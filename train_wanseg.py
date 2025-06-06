import os
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import torchvision
import diffusers
import math
import datetime
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    DDPMScheduler,
)
from wan.dataset import WanSegDataset
from wan.seg import WanSeg
from wan.text2video import FlowUniPCMultistepScheduler
from wan.configs import WAN_CONFIGS
from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs
from diffusers.utils import export_to_video


def parse_float_list(arg):
    try:
        return [float(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be floats")

def parse_int_list(arg):
    try:
        return [int(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List elements should be integers")

def parse_str_list(arg):
    return arg.split(',')

def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring the training session.
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()    

    # training details
    parser.add_argument("--output_dir", default='exp')
    parser.add_argument("--seed", type=int, default=41, help="A seed for reproducible training.")    
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=50000,)
    parser.add_argument("--global_step", type=int, default=0,)
    parser.add_argument("--checkpointing_steps", type=int, default=2500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--vace_learning_rate", type=float, default=1e-5, help="Learning rate for VACE layers")
    parser.add_argument("--prompt_learning_rate", type=float, default=5e-5, help="Learning rate for prompt encoder")
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-5, help="Learning rate for decoder")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")    
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of timesteps for the noise scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],)    
    parser.add_argument("--set_grads_to_none", action="store_true",)
    
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--dataloader_num_workers", type=int, default=16,)
    parser.add_argument("--logging_dir", type=str, default="logs")    
    parser.add_argument("--tracker_project_name", type=str, default="train_wanseg", help="The name of the wandb project to log to.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="The wandb entity/username to log to.")    

    parser.add_argument("--target_size", type=tuple, default=(512, 512))

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def gray_resize_for_identity(out, size=128):
    out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
    out_gray = out_gray.unsqueeze(1)
    out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
    return out_gray

def fix_state_dict(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    def print_gpu_memory(prefix=""):
        if accelerator.is_main_process:
            print(f"\n{prefix} GPU Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            print(f"Max Cached: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")

    arch_config = WAN_CONFIGS['t2v-1.3B']
    pipe = WanSeg(
        config=arch_config,
        checkpoint_dir='./Wan2.1-VACE-1.3B',
        pretrained_models_dir='./pretrained_models',
        target_size=args.target_size,
    )

    # Cast model to accelerator's dtype
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32  # default

    pipe = pipe.to(dtype=weight_dtype)
    pipe.model = pipe.model.to(dtype=weight_dtype)
    pipe.location_encoder = pipe.location_encoder.to(dtype=weight_dtype)
    pipe.vae = pipe.vae.to(dtype=weight_dtype)
    pipe.vae_project = pipe.vae_project.to(dtype=weight_dtype)
    pipe.context = pipe.context.to(dtype=weight_dtype)
    pipe.context_null = pipe.context_null.to(dtype=weight_dtype)

    # pipe.vae.enable_xformers_memory_efficient_attention()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Train prompt encoder, vace layers, and vae decoder
    vace_params, prompt_encoder_params, decoder_params = [], [], []
    cnt_vace, cnt_prompt_encoder, cnt_decoder = 0, 0, 0

    for name, param in pipe.model.named_parameters():
        if 'vace' in name:
            param.requires_grad_(True)
            vace_params.append(param)
            cnt_vace += param.numel()

    for name, param in pipe.location_encoder.named_parameters():
        param.requires_grad_(True)
        prompt_encoder_params.append(param)
        cnt_prompt_encoder += param.numel()
    
    for name, param in list(pipe.vae.model.decoder.named_parameters()) + list(pipe.vae_project.named_parameters()):
        param.requires_grad_(True)
        decoder_params.append(param)
        cnt_decoder += param.numel()    
    
    if accelerator.is_main_process:
        print('Trainable parameters:')
        print(f'Vace layers: {cnt_vace / 1e6}M')
        print(f'Prompt encoder: {cnt_prompt_encoder / 1e6}M')
        print(f'Decoder: {cnt_decoder / 1e6}M')
    
    # Create separate optimizers for each parameter group
    optimizer_vace = torch.optim.AdamW(vace_params, lr=args.vace_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    optimizer_prompt = torch.optim.AdamW(prompt_encoder_params, lr=args.prompt_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    optimizer_decoder = torch.optim.AdamW(decoder_params, lr=args.decoder_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)

    # Create separate schedulers for each optimizer
    lr_scheduler_vace = get_scheduler(args.lr_scheduler, optimizer=optimizer_vace,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    lr_scheduler_prompt = get_scheduler(args.lr_scheduler, optimizer=optimizer_prompt,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)
    lr_scheduler_decoder = get_scheduler(args.lr_scheduler, optimizer=optimizer_decoder,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)


    dataset = WanSegDataset(sources={
        'coco': './data/coco.jsonl',
    }, auto_init=True, target_size=args.target_size)

    total_size = len(dataset)
    valid_size = 10
    train_size = total_size - valid_size        

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Create dataloaders for each split
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    cnt_trainable, cnt_non_trainable = 0, 0
    for param in pipe.parameters():
        if param.requires_grad: 
            cnt_trainable += param.numel()
        else:
            cnt_non_trainable += param.numel()
    if accelerator.is_main_process:
        print(f"Trainable parameters: {cnt_trainable / 1e6}M")
        print(f"Non-trainable parameters: {cnt_non_trainable / 1e6}M")
    
    # Prepare everything with our `accelerator`.
    pipe, optimizer_vace, optimizer_prompt, optimizer_decoder, train_dataloader, valid_dataloader, lr_scheduler_vace, lr_scheduler_prompt, lr_scheduler_decoder = accelerator.prepare(
        pipe, optimizer_vace, optimizer_prompt, optimizer_decoder, train_dataloader, valid_dataloader, lr_scheduler_vace, lr_scheduler_prompt, lr_scheduler_decoder
    )

    # Initialize wandb
    if accelerator.is_main_process:
        runname = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

        if args.wandb_entity is not None:
            wandb.init(
                project=args.tracker_project_name,
                entity=args.wandb_entity,     
                name=runname,           
                config=vars(args),
            )
        else:
            wandb.init(
                project=args.tracker_project_name,
                name=runname,
                config=vars(args)
            )

    global_step = args.global_step
    progress_bar = tqdm(range(global_step, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process)
        
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(train_dataloader):            
            with accelerator.autocast(), accelerator.accumulate(pipe, optimizer_vace, optimizer_prompt, optimizer_decoder):
                image, gt_mask, points = batch
                # Move data to accelerator device
                image = image.to(accelerator.device)
                gt_mask = gt_mask.to(accelerator.device)[0]                

                points, labels = torch.stack(points[0]).unsqueeze(0).to(accelerator.device), points[1].unsqueeze(0).to(accelerator.device)
                points = points.transpose(-1, -2)

                # prepare x0
                mask_input = (gt_mask - 0.5) * 2
                mask_input = mask_input.unsqueeze(0).expand(3, -1, -1, -1)
                x0 = pipe.module.vae.encode([mask_input])[0].detach()

                # forward pass
                # Access model components through the module attribute when using DDP
                p0 = pipe.module.location_encoder(
                    points=(points, labels),
                    boxes=None,
                    masks=None,
                )[0]
                m0 = pipe.module.vae.encode([image[0]])[0].detach()

                target_shape = list(m0.shape)
                noise = torch.randn(
                    *target_shape,
                    dtype=weight_dtype,
                    device=accelerator.device
                )
                seq_len = math.ceil(((target_shape[2] * target_shape[3]) / 
                                    (pipe.module.patch_size[1] * pipe.module.patch_size[2])
                                    * target_shape[1]) / pipe.module.sp_size) * pipe.module.sp_size
                
                flow_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=args.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                    solver_order=1,
                )
                flow_scheduler.set_timesteps(args.num_train_timesteps)
                timestep = torch.randint(0, args.num_train_timesteps, (1,), device=accelerator.device)
                
                noisy_latents = flow_scheduler.add_noise(
                    x0,
                    noise,
                    timestep,
                )

                noise_pred = pipe.module.model(
                    [noisy_latents],
                    t=timestep,
                    vace_context=[p0, m0],
                    vace_context_scale=1.0,
                    context=pipe.module.context,
                    seq_len=seq_len
                )
                
                noise_pred = noise_pred
                noisy_latents = noisy_latents.unsqueeze(0)  # Add batch dimension
                latent_pred = flow_scheduler.step(
                    noise_pred,
                    timestep,
                    noisy_latents,
                    return_dict=False,
                )[0]
                latent_pred = latent_pred[0]
                latent_loss = F.mse_loss(latent_pred, x0, reduction="mean")
                
                decoded_latent = pipe.module.vae.decode([latent_pred])[0]
                mask_pred = pipe.module.vae_project(decoded_latent[:, 0])
                if mask_pred.shape != gt_mask.shape:
                    recovered_mask_pred = F.interpolate(mask_pred.unsqueeze(0), size=gt_mask.shape[-2:], mode='bilinear', align_corners=False)[0]
                else:
                    recovered_mask_pred = mask_pred
                decoder_loss = F.binary_cross_entropy_with_logits(recovered_mask_pred, gt_mask)                

                loss = latent_loss + decoder_loss
                loss = loss.to(dtype=weight_dtype)

                # backward pass
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vace_params, args.max_grad_norm)
                    accelerator.clip_grad_norm_(prompt_encoder_params, args.max_grad_norm)
                    accelerator.clip_grad_norm_(decoder_params, args.max_grad_norm)
                
                optimizer_vace.step()
                optimizer_prompt.step()
                optimizer_decoder.step()
                lr_scheduler_vace.step()
                lr_scheduler_prompt.step()
                lr_scheduler_decoder.step()
                optimizer_vace.zero_grad(set_to_none=args.set_grads_to_none)
                optimizer_prompt.zero_grad(set_to_none=args.set_grads_to_none)
                optimizer_decoder.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {
                        "latent_loss": latent_loss.detach().item(),
                        "decoder_loss": decoder_loss.detach().item(),
                        "total_loss": loss.detach().item(),
                        "lr_vace": lr_scheduler_vace.get_last_lr()[0],
                        "lr_prompt": lr_scheduler_prompt.get_last_lr()[0],
                        "lr_decoder": lr_scheduler_decoder.get_last_lr()[0],
                    }
                    progress_bar.set_postfix(**logs)

                    # Log to wandb
                    if args.report_to == "wandb":
                        wandb.log(logs, step=global_step)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 0:
                        os.makedirs(os.path.join(args.output_dir, runname,"checkpoints"), exist_ok=True)
                        outf = os.path.join(args.output_dir, runname, "checkpoints", f"model_{global_step}.pth")
                        torch.save(accelerator.unwrap_model(pipe).state_dict(), outf)
                        
                        # Run validation
                        if accelerator.is_main_process:
                            print("\nRunning validation...")
                            eval_pipe = accelerator.unwrap_model(pipe)
                            eval_pipe.eval()

                            val_out_dir = os.path.join(args.output_dir, runname, "eval", f"step_{global_step}")
                            os.makedirs(val_out_dir, exist_ok=True)

                            with torch.no_grad():
                                for val_idx, val_batch in enumerate(valid_dataloader):
                                    val_image, val_gt_mask, val_points = val_batch
                                    val_image = val_image.to(accelerator.device)[0]
                                    val_points, val_labels = torch.stack(val_points[0]).unsqueeze(0).to(accelerator.device), val_points[1].unsqueeze(0).to(accelerator.device)
                                    val_points = val_points.transpose(-1, -2)
                                    
                                    # Generate mask
                                    val_mask = eval_pipe(
                                        input_image=val_image,
                                        input_points=(val_points, val_labels),
                                        output_format='pil',
                                    )
                                                                                                            
                                    # Save input image                                    
                                    input_img = val_image[:, 0].cpu().numpy()
                                    input_img = ((input_img + 1) * 127.5).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)                                    
                                    Image.fromarray(input_img).save(os.path.join(val_out_dir, f"input_{val_idx}.png"))
                                    
                                    # Save ground truth mask
                                    gt_mask = val_gt_mask[0][0].cpu().numpy()
                                    gt_mask = (gt_mask * 255).clip(0, 255).astype(np.uint8)                                    
                                    Image.fromarray(gt_mask).save(os.path.join(val_out_dir, f"gt_mask_{val_idx}.png"))
                                    
                                    # Save predicted mask
                                    val_mask.save(os.path.join(val_out_dir, f"pred_mask_{val_idx}.png"))                                                                    
                            
                            eval_pipe.train()
                            print("Validation complete!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
