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
from .modules.maskdecoder import MaskDecoder3D
from .modules.coord_encoder import CoordinateConditionEncoder


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
    torch.save(context, save_path)
    return


def prepare_inputs(        
    input_image,
    input_points,
    target_size,
):
    input_area = input_image.size[0] * input_image.size[1]
    rescale_ratio = (target_size[0] * target_size[1] / input_area) ** 0.5
    rescaled_w = int(input_image.size[0] * rescale_ratio) // 16 * 16
    rescaled_h = int(input_image.size[1] * rescale_ratio) // 16 * 16
    input_image = input_image.convert('RGB').resize((rescaled_w, rescaled_h))
    input_image = TF.to_tensor(input_image).sub_(0.5).div_(0.5).unsqueeze(1)
    input_points = input_points[0], input_points[1]
    return input_image, input_points


class WanSeg(nn.Module):
    def __init__(
        self,
        config,
        checkpoint_dir,
        pretrained_models_dir,
        target_size,        
        device_id=None,        
    ):
        super().__init__()
        if device_id is None:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device(f"cuda:{device_id}")
        
        self.config = config
        self.num_train_timesteps = config.num_train_timesteps        

        # self.location_encoder = LocationEncoder(embed_dim=256, output_dim=1536)
        # self.location_encoder.to(self.device)
        self.coord_encoder = CoordinateConditionEncoder()
        self.coord_encoder.to(self.device)
 
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device
        )
        del self.vae.model.decoder
        self.vae.model.decoder = MaskDecoder3D(in_channels=16)
        self.vae.model.decoder.load_state_dict(torch.load(os.path.join(pretrained_models_dir, 'maskdecoder.pth')))
        self.vae.model.decoder.to(self.device)
        
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.vace_patch_embedding = nn.Conv3d(
            36, # was 96
            self.model.dim,
            kernel_size=self.model.patch_size,
            stride=self.model.patch_size
        )
        self.model.squeeze_ffn(ratio=2)
        # self.model = VaceWanModel()
        self.model.eval().requires_grad_(False)

        self.context = torch.load(os.path.join(pretrained_models_dir, 'segment_prompt.pt'))
        self.context = nn.ParameterList([nn.Parameter(x.to(self.device), requires_grad=False) for x in self.context])
        self.context_null = torch.load(os.path.join(pretrained_models_dir, 'segment_negative_prompt.pt'))
        self.context_null = nn.ParameterList([nn.Parameter(x.to(self.device), requires_grad=False) for x in self.context_null])

        self.sp_size = 1
        self.model.to(self.device)
        self.sample_neg_prompt = config.sample_neg_prompt

        self.target_size = target_size

    def prepare_input(
        self,
        input_image,
        input_points,        
    ):
        input_image, input_points = prepare_inputs(
            input_image,
            input_points,
            target_size=self.target_size,
        )
        input_image = input_image.to(self.device)
        input_points = input_points[0].to(self.device), input_points[1].to(self.device)        

        return input_image, input_points


    def forward(
        self,
        input_image,
        input_points=None,
        shift=5.0,
        sampling_steps=50,
        context_scale=1.0,
        # sample_solver='unipc',        
        # guide_scale=5.0,
        # prompt="",
        # negative_prompt="",
        seed=-1,
        output_format='pil',
    ):
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)        

        # p0 = self.location_encoder(
        #     input_points,
        #     boxes=None,
        #     masks=None,
        # )[0]
        m0 = self.vae.encode([input_image])        
        coords, labels = input_points                
        p0 = self.coord_encoder(coords[:, 0, :], m0[0].shape[-2], m0[0].shape[-1])
        p0 = p0.unsqueeze(1)
        print(coords[:, 0, :])

        vace_context = torch.cat([p0, m0[0]])

        target_shape = list(m0[0].shape)
        noise = [torch.randn(*target_shape, dtype=p0.dtype, device=self.device, generator=seed_g)]
        seq_len = math.ceil(((target_shape[2] * target_shape[3]) / 
                             (self.patch_size[1] * self.patch_size[2])
                             * target_shape[1]) / self.sp_size) * self.sp_size
        # seq_len += 2

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)
        
        with torch.amp.autocast(device_type='cuda', dtype=p0.dtype), torch.no_grad(), no_sync():
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                solver_order=3,
                use_dynamic_shifting=False
            )
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps

            latents = noise
            arg_c = {'context': self.context, 'seq_len': seq_len}
            
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)

                noise_pred = self.model(
                    latent_model_input,
                    t=timestep,
                    vace_context=[vace_context],
                    vace_context_scale=context_scale,
                    **arg_c
                )[0]

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g
                )[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
        
        output_image = self.vae.model.decoder(x0[0].unsqueeze(0))[0][0]
        output_image = torch.where(output_image > 0, 255, 0)
        if output_format == 'pil':
            output_image = Image.fromarray(output_image[0].detach().cpu().numpy().astype(np.uint8))
        
        return output_image
