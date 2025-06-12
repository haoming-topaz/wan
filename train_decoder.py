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
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--global_step", type=int, default=0,)
    parser.add_argument("--checkpointing_steps", type=int, default=10000,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)    
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
    parser.add_argument("--tracker_project_name", type=str, default="train_mask_decoder", help="The name of the wandb project to log to.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="The wandb entity/username to log to.")    

    parser.add_argument("--target_size", type=tuple, default=(768, 768))

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
        os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    

    arch_config = WAN_CONFIGS['t2v-1.3B']
    pipe = WanSeg(
        config=arch_config,
        checkpoint_dir='./Wan2.1-VACE-1.3B',
        pretrained_models_dir='./pretrained_models',
        target_size=args.target_size,
    )
    vae = pipe.vae

    # Cast model to accelerator's dtype
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32  # default

    vae = vae.to(dtype=weight_dtype)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Train prompt encoder, vace layers, and vae decoder
    decoder_params = []
    cnt_decoder = 0
    
    for name, param in vae.model.decoder.named_parameters():
        param.requires_grad_(True)
        decoder_params.append(param)
        cnt_decoder += param.numel()
    
    if accelerator.is_main_process:
        print('Trainable parameters:')
        print(f'Decoder: {cnt_decoder / 1e6}M')
        
    optimizer_decoder = torch.optim.AdamW(decoder_params, lr=args.decoder_learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
        
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

    if accelerator.is_main_process:
        cnt_trainable, cnt_non_trainable = 0, 0
        for name,param in vae.named_parameters():
            if param.requires_grad:
                print(name)
                cnt_trainable += param.numel()
            else:
                cnt_non_trainable += param.numel()
        print(f"Trainable parameters: {cnt_trainable / 1e6}M")
        print(f"Non-trainable parameters: {cnt_non_trainable / 1e6}M")
    
    # Prepare everything with our `accelerator`.
    vae, optimizer_decoder, train_dataloader, lr_scheduler_decoder = accelerator.prepare(
        vae, optimizer_decoder, train_dataloader, lr_scheduler_decoder
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
    
    for _ in range(0, args.num_training_epochs):
        for batch in train_dataloader:            
            with accelerator.autocast(), accelerator.accumulate(vae, optimizer_decoder):
                _, gt_mask, points, _ = batch
                # Move data to accelerator device
                gt_mask = gt_mask.to(accelerator.device)

                # prepare x0
                mask_input = gt_mask.expand(3, -1, -1, -1)
                x0 = vae.module.encode([mask_input])[0].detach()
                pred_mask = vae.module.model.decoder(x0.unsqueeze(0))[0]
                gt_mask = torch.where(gt_mask > 0, torch.ones_like(gt_mask, device=accelerator.device), torch.zeros_like(gt_mask, device=accelerator.device))
                
                num_positive = gt_mask.sum()
                num_negative = (gt_mask == 0).sum()
                pos_weight_value = num_negative / (num_positive + 1e-6)

                bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
                bce_loss = bce_loss_fn(pred_mask, gt_mask)

                probs = torch.sigmoid(pred_mask)
                intersection = (probs * gt_mask).sum()
                dice_score = (2. * intersection + 1e-6) / (probs.sum() + num_positive + 1e-6)
                dice_loss = 1 - dice_score

                lambda_dice = 0.5
                decoder_loss = bce_loss + lambda_dice * dice_loss

                # backward pass
                accelerator.backward(decoder_loss)

                if accelerator.sync_gradients:                    
                    accelerator.clip_grad_norm_(decoder_params, args.max_grad_norm)
                                
                optimizer_decoder.step()                
                lr_scheduler_decoder.step()                
                optimizer_decoder.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step >= args.max_train_steps:
                    break  # break out of inner loop

                if accelerator.is_main_process:
                    logs = {                        
                        "decoder_loss": decoder_loss.detach().item(),
                        "bce_loss": bce_loss.detach().item(),
                        "dice_loss": dice_loss.detach().item(),                        
                    }
                    progress_bar.set_postfix(**logs)

                    # Log to wandb
                    if args.report_to == "wandb":
                        wandb.log(logs, step=global_step)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 2:
                        os.makedirs(os.path.join(args.output_dir, runname, "checkpoints"), exist_ok=True)
                        outf = os.path.join(args.output_dir, runname, "checkpoints", f"decoder_{global_step}.pth")
                        eval_vae = accelerator.unwrap_model(vae)
                        torch.save(eval_vae.model.decoder.state_dict(), outf)
                        
                        # Run validation
                        print("\nRunning validation...")                        
                        eval_vae.eval()

                        val_out_dir = os.path.join(args.output_dir, runname, "eval", f"step_{global_step}")
                        os.makedirs(val_out_dir, exist_ok=True)

                        with torch.no_grad():
                            for val_idx, val_batch in enumerate(valid_dataloader):
                                _, val_mask, _, paths = val_batch

                                mask_input = val_mask.expand(3, -1, -1, -1).to(accelerator.device)
                                x0 = eval_vae.encode([mask_input])[0].detach()
                                pred_mask = eval_vae.model.decoder(x0.unsqueeze(0))[0]
                                pred_mask = pred_mask[0][0]
                                pred_mask = torch.where(
                                    pred_mask > 0,
                                    torch.ones_like(pred_mask, dtype=torch.uint8),
                                    torch.zeros_like(pred_mask, dtype=torch.uint8),
                                )
                                pred_mask = pred_mask.cpu().numpy() * 255
                                pred_mask = Image.fromarray(pred_mask.astype(np.uint8))
                                                                                                            
                                # Save input
                                Image.open(paths[0][0]).save(os.path.join(val_out_dir, f"input_{val_idx}.png"))
                                Image.open(paths[1][0]).save(os.path.join(val_out_dir, f"gt_mask_{val_idx}.png"))
                                # Save predicted mask
                                pred_mask.save(os.path.join(val_out_dir, f"pred_mask_{val_idx}.png"))                                                                    
                        
                        vae.train()
                        print("Validation complete!")

        # Add a check to break the outer loop as well
        if global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
