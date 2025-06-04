import gc
import logging
import math
import os
import random
import sys
import time
import traceback
import types
from contextlib import contextmanager
from functools import partial
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn
from PIL import Image
from tqdm import tqdm
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.location_encoder import LocationEncoder
from .text2video import (
    FlowDPMSolverMultistepScheduler,
    FlowUniPCMultistepScheduler,
    T5EncoderModel,
    WanT2V,
    WanVAE,
    get_sampling_sigmas,
    retrieve_timesteps,
    shard_model,
)
from .vace import WanVace
from .modules.vace_model import VaceWanModel


def prepare_text_input(config, checkpoint_dir, text, save_path):
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        shard_fn=None,
    )
    text_encoder.model.eval().requires_grad_(False).to('cuda')
    context = text_encoder([text], torch.device('cuda'))
    print(context)
    print(context[0].shape)
    torch.save(context, save_path)
    return


class WanSeg(WanVace):
    def __init__(
        self,
        config,
        checkpoint_dir,
        pretrained_models_dir,
        device_id=None,
        rank=None,
    ):
        if device_id is not None:
            self.device = torch.device(f"cuda:{device_id}")
            self.rank = rank
        else:
            self.device = torch.device('cpu')
            self.rank = 0
        
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.location_encoder = LocationEncoder(embed_dim=1024, output_dim=4096)
        self.location_encoder.to(self.device)
 
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device
        )

        logging.info(f"Creating WanSegModel from {checkpoint_dir}")
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.vace_patch_embedding = nn.Conv3d(
            16,
            self.model.dim,
            kernel_size=self.model.patch_size,
            stride=self.model.patch_size
        )
        self.model.eval().requires_grad_(False)

        self.context = [torch.load(os.path.join(pretrained_models_dir, 'segment_prompt.pt'))]
        self.context_null = [torch.load(os.path.join(pretrained_models_dir, 'segment_negative_prompt.pt'))]

        self.sp_size = 1
        self.model.to(self.device)
        self.sample_neg_prompt = config.sample_neg_prompt

    def prepare_inputs(
        self,
        input_image,
        input_points=None,
        target_size=(768, 768),
    ):
        input_area = input_image.size[0] * input_image.size[1]
        rescale_ratio = (target_size[0] * target_size[1] / input_area) ** 0.5
        rescaled_w = int(input_image.size[0] * rescale_ratio) // 16 * 16
        rescaled_h = int(input_image.size[1] * rescale_ratio) // 16 * 16
        input_image = input_image.convert('RGB').resize((rescaled_w, rescaled_h))
        input_image = TF.to_tensor(input_image).sub_(0.5).div_(0.5).unsqueeze(1).to(self.device)
        input_points = input_points[0].to(self.device), input_points[1].to(self.device)
        return input_image, input_points

    def generate(
        self,
        input_image,
        input_points=None,
        size=(1280, 720),
        shift=5.0,
        sample_solver='unipc',
        sampling_steps=25,
        guide_scale=5.0,
        context_scale=1.0,
        # prompt="",
        # negative_prompt="",
        max_area=720 * 1280,
        seed=-1,
    ):
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        input_image, input_points = self.prepare_inputs(
            input_image,
            input_points,
        )

        context = self.location_encoder(
            input_points,
            boxes=None,
            masks=None,
        )[0]
        print(context.shape)

        m0 = self.vae.encode([input_image])
        print(m0[0].shape)

        target_shape = list(m0[0].shape)
        noise = [torch.randn(*target_shape, dtype=torch.float32, device=self.device, generator=seed_g)]

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        print(seq_len)
        with torch.amp.autocast(device_type='cuda', dtype=self.param_dtype), torch.no_grad(), no_sync():
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False
            )
            print(self.num_train_timesteps)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)
                self.model.to(self.device)
                noise_pred = self.model(
                    latent_model_input,
                    t=timestep,
                    vace_context=m0,
                    vace_context_scale=context_scale,
                    **arg_c)[0]

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g
                )[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
        
        if self.rank == 0:
            videos = self.vae.decode(x0)
        
        return videos[0]
