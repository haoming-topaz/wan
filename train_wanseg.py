import os
import gc
import random
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
from PIL import Image, ImageDraw
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
from diffusers.training_utils import (
    EMAModel,
    compute_loss_weighting_for_sd3,
    compute_density_for_timestep_sampling
)


def parse_args(input_args=None):
    """
    Parses command-line arguments used for configuring the training session.
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # training details
    parser.add_argument("--output_dir", default='exp-wan')
    parser.add_argument("--seed", type=int, default=41, help="A seed for reproducible training.")    
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=50005,)
    parser.add_argument("--global_step", type=int, default=0,)
    parser.add_argument("--checkpointing_steps", type=int, default=2500,)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
        help="Path to a specific checkpoint to resume training from. If None, training will start from scratch.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--vace_learning_rate", type=float, default=1e-5, help="Learning rate for VACE layers")
    parser.add_argument("--prompt_learning_rate", type=float, default=2e-5, help="Learning rate for prompt encoder")
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
    
    parser.add_argument("--anchor_warmup_steps", type=int, default=1000)
    parser.add_argument("--anchor_loss_lambda", type=float, default=0.5)
    parser.add_argument("--curriculum_warmup_steps", type=int, default=5000)
    parser.add_argument("--flow_velocity_norm_lambda", type=float, default=0)
    parser.add_argument("--multi_step_increase_steps", type=int, default=5000)
    parser.add_argument("--multi_step_maximum", type=int, default=3)
    parser.add_argument("--pure_noise_warmup_steps", type=int, default=10000)
    parser.add_argument("--pure_noise_prob", type=float, default=0.0)

    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--dataloader_num_workers", type=int, default=16,)
    parser.add_argument("--logging_dir", type=str, default="logs")    
    parser.add_argument("--tracker_project_name", type=str, default="train_wanseg", help="The name of the wandb project to log to.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="The wandb entity/username to log to.")    

    parser.add_argument("--target_size", type=int, nargs=2, default=(512, 512))

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def wrap_state_dict(state_dict):
    return {'module.' + k: v for k, v in state_dict.items()}


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
        os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    
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
    pipe.coord_encoder = pipe.coord_encoder.to(dtype=weight_dtype)
    pipe.vae = pipe.vae.to(dtype=weight_dtype)
    pipe.context = pipe.context.to(dtype=weight_dtype)
    pipe.context_null = pipe.context_null.to(dtype=weight_dtype)

    # pipe.vae.enable_xformers_memory_efficient_attention()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Train prompt encoder, vace layers, and vae decoder
    vace_params, prompt_encoder_params = [], []
    cnt_vace, cnt_prompt_encoder = 0, 0

    for param in pipe.parameters():
        param.requires_grad_(False)

    for name, param in pipe.model.named_parameters():
        if 'vace' in name:
            param.requires_grad_(True)
            vace_params.append(param)
            cnt_vace += param.numel()

    for name, param in pipe.coord_encoder.named_parameters():
        param.requires_grad_(True)
        prompt_encoder_params.append(param)
        cnt_prompt_encoder += param.numel()
    
    if accelerator.is_main_process:
        print('Trainable parameters:')
        print(f'Vace layers: {cnt_vace / 1e6}M')
        print(f'Prompt encoder: {cnt_prompt_encoder / 1e6}M')
    
    # Create separate optimizers for each parameter group
    optimizer_vace = torch.optim.AdamW(vace_params, lr=args.vace_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    optimizer_prompt = torch.optim.AdamW(prompt_encoder_params, lr=args.prompt_learning_rate,
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

    if accelerator.is_main_process:
        cnt_trainable, cnt_non_trainable = 0, 0
        for name, param in pipe.named_parameters():
            if param.requires_grad:
                print(name)
                cnt_trainable += param.numel()
            else:
                cnt_non_trainable += param.numel()
        print(f"Trainable parameters: {cnt_trainable / 1e6}M")
        print(f"Non-trainable parameters: {cnt_non_trainable / 1e6}M")
    
    # Prepare everything with our `accelerator`.
    pipe, optimizer_vace, optimizer_prompt, train_dataloader, lr_scheduler_vace, lr_scheduler_prompt = accelerator.prepare(
        pipe, optimizer_vace, optimizer_prompt, train_dataloader, lr_scheduler_vace, lr_scheduler_prompt
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

    # Load checkpoint if specified
    if args.resume_from_checkpoint is not None:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        # Load the checkpoint
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        
        # Load model state
        model_state_dict = wrap_state_dict(checkpoint["model_state_dict"])
        pipe.load_state_dict(model_state_dict)
        
        # Load optimizer states
        optimizer_vace.load_state_dict(checkpoint["optimizer_vace_state_dict"])
        optimizer_prompt.load_state_dict(checkpoint["optimizer_prompt_state_dict"])
        
        # Load scheduler states
        lr_scheduler_vace.load_state_dict(checkpoint["lr_scheduler_vace_state_dict"])
        lr_scheduler_prompt.load_state_dict(checkpoint["lr_scheduler_prompt_state_dict"])
        
        # Update global step
        global_step = checkpoint["global_step"]
        
        if accelerator.is_main_process:
            print(f"Resumed from step {global_step}")

    progress_bar = tqdm(range(global_step, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process)
    
    pipe.train()
    multi_step_count = 1
    for _ in range(0, args.num_training_epochs):
        for batch in train_dataloader:
            with accelerator.autocast(), accelerator.accumulate(pipe, optimizer_vace, optimizer_prompt):
                image, gt_mask, points, _ = batch
                # Move data to accelerator device
                image = image.to(accelerator.device)
                gt_mask = gt_mask.to(accelerator.device)                

                points, labels = torch.stack(points[0]).unsqueeze(0).to(accelerator.device), points[1].unsqueeze(0).to(accelerator.device)
                points = points.transpose(-1, -2)                

                # prepare x0
                mask_input = gt_mask.expand(3, -1, -1, -1)
                x0 = pipe.module.vae.encode([mask_input])[0].detach()

                # forward pass
                # Access model components through the module attribute when using DDP
                # p0 = pipe.module.location_encoder(
                #     points=(points, labels),
                #     boxes=None,
                #     masks=None,
                # )[0]

                m0 = pipe.module.vae.encode([image[0]])[0].detach()
                p0 = pipe.module.coord_encoder(points[0], m0.shape[-2], m0.shape[-1])
                p0 = p0.unsqueeze(1)
                vace_context = torch.cat([p0, m0])                

                flow_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=args.num_train_timesteps,
                    shift=1,                    
                    use_dynamic_shifting=False,
                )
                flow_scheduler.set_timesteps(args.num_train_timesteps)

                target_shape = list(m0.shape)
                noise = torch.randn(
                    *target_shape,
                    dtype=weight_dtype,
                    device=accelerator.device
                )
                seq_len = math.ceil(((target_shape[2] * target_shape[3]) / 
                                    (pipe.module.patch_size[1] * pipe.module.patch_size[2])
                                    * target_shape[1]) / pipe.module.sp_size) * pipe.module.sp_size
                # seq_len += 2

                curriculum_progress = min(1.0, global_step / args.curriculum_warmup_steps)
                upper_bound = min(args.num_train_timesteps - 1, int(curriculum_progress * args.num_train_timesteps))
                upper_bound = max(10, upper_bound)
                
                step_idx = random.randint(0, upper_bound - multi_step_count)
                timestep = torch.stack([flow_scheduler.timesteps[step_idx]]).to(accelerator.device)
                
                use_pure_noise = False
                # if global_step > args.pure_noise_warmup_steps:
                #     random_noise_prob = min(args.pure_noise_prob, args.pure_noise_prob * (global_step - args.pure_noise_warmup_steps) / args.pure_noise_warmup_steps)
                #     if random.random() < random_noise_prob:
                #         use_pure_noise = True

                if use_pure_noise:
                    noisy_latents = torch.randn_like(x0)
                else:
                    sigmas = flow_scheduler.sigmas.to(accelerator.device)                    
                    sigmas = sigmas[step_idx]
                    noisy_latents = (1.0 - sigmas) * x0 + sigmas * noise
                    # noisy_latents = flow_scheduler.add_noise(x0, noise, timestep)                

                target = noise - x0

                latent_losses = []
                # anchor_losses = []
                # velocity_losses = []

                for K in range(multi_step_count):
                    noise_pred = pipe.module.model(
                        [noisy_latents],
                        t=timestep,
                        vace_context=[vace_context],
                        vace_context_scale=1.0,
                        context=pipe.module.context,
                        seq_len=seq_len
                    )[0]

                    latent_losses.append(F.mse_loss(noise_pred, target, reduction='mean'))
                    
                    # current_latent = flow_scheduler.step(
                    #     target,
                    #     timestep,
                    #     noisy_latents,
                    # )[0]

                    # if accelerator.is_main_process:
                    #     print('x0', accelerator.device, x0[0][0])
                    #     print('noise', accelerator.device, noise[0][0])
                    #     print('noisy_latents', accelerator.device, noisy_latents[0][0])
                    #     print('current_latent', accelerator.device, current_latent[0][0])
                    #     assert False
                    
                    # anchor_losses.append(F.mse_loss(current_latent, x0, reduction='mean'))
                    # velocity_losses.append(noise_pred.norm())

                    # noisy_latents = current_latent
                    if K + 1 < multi_step_count:
                        timestep = flow_scheduler.timesteps[flow_scheduler.step_index]
                
                latent_loss = sum(latent_losses) / len(latent_losses)
                # Anchor loss
                # anchor_loss = sum(anchor_losses) / len(anchor_losses)
                # lambda_anchor_loss = min(args.anchor_loss_lambda, global_step / args.anchor_warmup_steps)                                
                # velocity_loss = sum(velocity_losses) / len(velocity_losses)
                
                loss = latent_loss # + \
                    # lambda_anchor_loss * anchor_loss + \
                    # args.flow_velocity_norm_lambda * velocity_loss
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vace_params, args.max_grad_norm)
                    accelerator.clip_grad_norm_(prompt_encoder_params, args.max_grad_norm)
                
                optimizer_vace.step()
                optimizer_prompt.step()
                lr_scheduler_vace.step()
                lr_scheduler_prompt.step()
                optimizer_vace.zero_grad(set_to_none=args.set_grads_to_none)
                optimizer_prompt.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process:
                    logs = {
                        "loss": loss.detach().item(),                        
                        # "latent_loss": latent_loss.detach().item(),
                        # '1-step latent loss': latent_losses[-1].detach().item(),
                        # '2-step latent loss': 0 if len(latent_losses) < 2 else latent_losses[-2].detach().item(),
                        # '3-step latent loss': 0 if len(latent_losses) < 3 else latent_losses[-3].detach().item(),
                        # "anchor_loss": anchor_loss.detach().item(),
                        # '1-step anchor loss': anchor_losses[-1].detach().item(),
                        # '2-step anchor loss': 0 if len(anchor_losses) < 2 else anchor_losses[-2].detach().item(),
                        # '3-step anchor loss': 0 if len(anchor_losses) < 3 else anchor_losses[-3].detach().item(),
                        # "velocity_loss": velocity_loss.detach().item(),
                        # "curriculum_progress": curriculum_progress,
                        # "K": multi_step_count,
                    }
                    progress_bar.set_postfix(**logs)

                    if global_step % args.multi_step_increase_steps == 0 and multi_step_count < args.multi_step_maximum:
                        multi_step_count += 1

                    # Log to wandb
                    if args.report_to == "wandb":
                        wandb.log(logs, step=global_step)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 2 or global_step == args.max_train_steps:
                        os.makedirs(os.path.join(args.output_dir, runname, "checkpoints"), exist_ok=True)
                        outf = os.path.join(args.output_dir, runname, "checkpoints", f"model_{global_step}.pth")
                        if global_step >= 5000:
                            # Save full checkpoint
                            checkpoint = {
                                "model_state_dict": accelerator.unwrap_model(pipe).state_dict(),
                                "optimizer_vace_state_dict": optimizer_vace.state_dict(),
                                "optimizer_prompt_state_dict": optimizer_prompt.state_dict(),
                                "lr_scheduler_vace_state_dict": lr_scheduler_vace.state_dict(),
                                "lr_scheduler_prompt_state_dict": lr_scheduler_prompt.state_dict(),
                                "global_step": global_step,
                            }
                            torch.save(checkpoint, outf)
                        
                        # Run validation
                        print("\nRunning validation...")
                        eval_pipe = accelerator.unwrap_model(pipe)
                        eval_pipe.eval()

                        val_out_dir = os.path.join(args.output_dir, runname, "eval", f"step_{global_step}")
                        os.makedirs(val_out_dir, exist_ok=True)

                        with torch.no_grad():
                            for val_idx, val_batch in enumerate(valid_dataloader):
                                val_image, val_gt_mask, val_points, paths = val_batch
                                val_image = val_image.to(accelerator.device)[0]
                                val_points, val_labels = torch.stack(val_points[0]).unsqueeze(0).to(accelerator.device), val_points[1].unsqueeze(0).to(accelerator.device)
                                val_points = val_points.transpose(-1, -2)
                                
                                # Generate mask
                                val_mask = eval_pipe(
                                    input_image=val_image,
                                    input_points=(val_points, val_labels),
                                    output_format='pil',                                    
                                )

                                mask_input = val_gt_mask.expand(3, -1, -1, -1).to(accelerator.device)
                                x0 = eval_pipe.vae.encode([mask_input])[0].detach()                                
                                
                                pred_mask = eval_pipe.vae.model.decoder(x0.unsqueeze(0))[0]
                                pred_mask = pred_mask[0][0]
                                pred_mask = torch.where(
                                    pred_mask > 0,
                                    torch.ones_like(pred_mask, dtype=torch.uint8),
                                    torch.zeros_like(pred_mask, dtype=torch.uint8),
                                )
                                pred_mask = pred_mask.cpu().numpy() * 255
                                pred_mask = Image.fromarray(pred_mask.astype(np.uint8)).convert('RGB')
                                draw = ImageDraw.Draw(pred_mask)
                                
                                refer_point = list(val_points[0][0].detach().cpu().numpy())
                                x, y = refer_point
                                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red')
                                pred_mask.save(os.path.join(val_out_dir, f"recovered_mask_{val_idx}.png"))

                                # Save input
                                Image.open(paths[0][0]).save(os.path.join(val_out_dir, f"input_{val_idx}.png"))
                                Image.open(paths[1][0]).save(os.path.join(val_out_dir, f"gt_mask_{val_idx}.png"))
                                # Save predicted mask
                                val_mask.save(os.path.join(val_out_dir, f"pred_mask_{val_idx}.png"))                                                                    
                        
                        eval_pipe.train()
                        print("Validation complete!")

                if global_step >= args.max_train_steps:
                    break
            
            if global_step >= args.max_train_steps:
                break
                
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
