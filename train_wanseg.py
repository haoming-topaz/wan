import os
import gc
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import torchvision
import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler
from wan.dataset import WanSegDataset
from wan.seg import WanSeg
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
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)

    # training details
    parser.add_argument("--output_dir", default='experience/osediff')
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=100000,)
    parser.add_argument("--global_step", type=int, default=0,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)    
    parser.add_argument("--vace_learning_rate", type=float, default=5e-5, help="Learning rate for VACE layers")
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

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument("--logging_dir", type=str, default="logs")
    
    
    parser.add_argument("--tracker_project_name", type=str, default="train_osediff", help="The name of the wandb project to log to.")
    parser.add_argument('--dataset_txt_paths_list', type=parse_str_list, default=['YOUR TXT FILE PATH'], help='A comma-separated list of integers')
    parser.add_argument('--dataset_prob_paths_list', type=parse_int_list, default=[1], help='A comma-separated list of integers')
    parser.add_argument("--deg_file_path", default="params_realesrgan.yml", type=str)
    parser.add_argument("--pretrained_model_name_or_path", default=None, type=str)
    parser.add_argument("--merged_unet_vae", default=None, type=str)
    parser.add_argument("--d_init_path", default=None, type=str)
    parser.add_argument('--d_safetensor', type=str, default=None, help='Path to RAM model')
    parser.add_argument('--g_safetensor', type=str, default=None, help='Path to RAM model')

    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_latent", default=1.0, type=float)
    parser.add_argument("--lambda_motion", default=0.0, type=float)
    parser.add_argument("--lambda_lpips", default=2.0, type=float)
    parser.add_argument("--lambda_g", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--neg_prompt", default="", type=str)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)
    # parser.add_argument("--spatial_size", default=32, type=int)
    
    parser.add_argument("--use_generic_prompt", action="store_true",)
    parser.add_argument("--use_null_prompt", action="store_true",)
    parser.add_argument("--image_condition_unet", default=None, type=str)
    parser.add_argument("--deblur", action="store_true",)
    parser.add_argument("--learn_residue", action="store_true",)
    parser.add_argument("--gan_type", type=str, default="vanilla", choices=["vanilla", "wgan_softplus", "wgan", "hinge", "lsgan"],)
    parser.add_argument("--no_degradation", action="store_true",)
    # lora setting
    parser.add_argument("--lora_rank", default=4, type=int)
    parser.add_argument("--d_lora_rank", default=4, type=int)
    parser.add_argument("--num_frames", default=21, type=int)
    # ram path
    parser.add_argument('--ram_path', type=str, default=None, help='Path to RAM model')
    parser.add_argument("--dust", action="store_true",)


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

    config = WAN_CONFIGS['t2v-1.3B']
    pipe = WanSeg(
        config=config,
        checkpoint_dir='./Wan2.1-VACE-1.3B',
    )

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
    
    for name, param in pipe.vae.model.decoder.named_parameters():
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


    dataset_train = WanSegDataset({
        'coco': './data/coco.jsonl',
    })
    
    # Prepare everything with our `accelerator`.
    model_gen, model_reg, optimizer_vace, optimizer_prompt, optimizer_decoder, optimizer_reg, dl_train, lr_scheduler_vace, lr_scheduler_prompt, lr_scheduler_decoder, lr_scheduler_reg = accelerator.prepare(
        model_gen, model_reg, optimizer_vace, optimizer_prompt, optimizer_decoder, optimizer_reg, dl_train, lr_scheduler_vace, lr_scheduler_prompt, lr_scheduler_decoder, lr_scheduler_reg
    )
    net_lpips = accelerator.prepare(net_lpips)
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        args.dataset_txt_paths_list = str(args.dataset_txt_paths_list)
        args.dataset_prob_paths_list = str(args.dataset_prob_paths_list)
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    global_step = args.global_step
    progress_bar = tqdm(range(global_step, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):

            m_acc = [model_gen, model_reg]
            with accelerator.accumulate(*m_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]  # b, f, c, h, w 
                if args.dust:
                    mask = batch["mask"]

                # get text prompts from GT
                if args.use_generic_prompt:
                    batch_size = 1
                    batch["prompt"] = ["high quality, high re solution, details, sharp, photorealistic"] * batch_size
                else:
                    x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                    caption = inference(x_tgt_ram.to(dtype=torch.float16), model_vlm)
                    batch["prompt"] = [f'{each_caption}' for each_caption in caption]
                        
                x_src = x_src.unsqueeze(1)
                x_tgt = x_tgt.unsqueeze(1)
                
                # if batch["lq"].shape[1] != args.num_frames or batch["gt"].shape[1] != args.num_frames:
                #     lq_shape= batch["lq"].shape
                #     gt_shape= batch["gt"].shape
                #     print(f"Input data shape mismatch: {lq_shape} != {gt_shape}")

                #     padT = args.num_frames - lq_shape[1]
                #     padding = (0, 0, 0, 0, 0, int(padT))
                #     batch["lq"] = torch.nn.functional.pad(batch["lq"].permute(0, 2, 1, 3, 4), padding, mode="replicate").permute(0, 2, 1, 3, 4)
                #     batch["gt"] = torch.nn.functional.pad(batch["gt"].permute(0, 2, 1, 3, 4), padding, mode="replicate").permute(0, 2, 1, 3, 4)
                #     print(f"Padding data shape: {batch['lq'].shape} == {batch['gt'].shape}")

                #     # lq_dtype = batch["lq"].dtype
                #     # batch["lq"], batch["gt"] = 0, 0
                #     # batch["lq"] = torch.zeros([1, args.num_frames, 3, 480, 832]).to(accelerator.device).to(dtype=lq_dtype)
                #     # batch["gt"] = torch.zeros([1, args.num_frames, 3, 480, 832]).to(accelerator.device).to(dtype=lq_dtype)

                # x_src = batch["lq"] * 2 - 1 
                # x_tgt = batch["gt"] * 2 - 1

#                 # # degradation check before training            
#                 # x_tgt = x_tgt * 0.5 + 0.5
#                 # x_src = x_src * 0.5 + 0.5
#                 # x_tgt = x_tgt.clamp(0, 1)[0].permute(0, 2, 3, 1).detach().cpu().numpy()
#                 # x_src = x_src.clamp(0, 1)[0].permute(0, 2, 3, 1).detach().cpu().numpy()
#                 # export_to_video([l for l in x_tgt], f"preset/datasets/data_check/gt_{global_step}_{accelerator.device}.mp4", fps=16)
#                 # export_to_video([l for l in x_src], f"preset/datasets/data_check/lq_{global_step}_{accelerator.device}.mp4", fps=16)
        
#                 # global_step += 1
#                 # continue

                x_src = x_src.permute(0, 2, 1, 3, 4) # b, f, c, h, w --> b, c, f, h, w  # need to change
                x_tgt = x_tgt.permute(0, 2, 1, 3, 4)

#                 # print(x_src.shape, x_tgt.shape)

#                 # x_src = model_gen.module.random_sample(batch_size=1, num_frames=21)
#                 # x_tgt = model_gen.module.random_sample(batch_size=1, num_frames=21).to(accelerator.device)
#                 # combined = torch.concat([x_src, x_tgt], 2)
#                 # torchvision.utils.save_image(0.5*combined + 0.5, f'tb_{np.random.randint(0, 99999)}.png')
#                 # B, C, H, W = x_src.shape
                

                # forward pass
                x_tgt_pred, latents_pred, prompt_embeds, latents_tgt = model_gen(x_src, x_tgt, batch=batch, args=args)
                if args.lambda_latent > 0:
                    loss_latent = F.mse_loss(latents_pred.float(), latents_tgt.float(), reduction="mean") * args.lambda_latent
                else:
                    loss_latent = torch.zeros(1).to(x_src.device)

                if args.lambda_motion > 0:
                    x_tgt_pred_motion = x_tgt_pred.permute(0, 2, 1, 3, 4)
                    x_tgt_pred_motion = x_tgt_pred_motion.view(-1, 3, x_tgt_pred_motion.size(3), x_tgt_pred_motion.size(4))
                    
                    x_tgt_motion = x_tgt.permute(0, 2, 1, 3, 4)
                    x_tgt_motion = x_tgt_motion.view(-1, 3, x_tgt_motion.size(3), x_tgt_motion.size(4))
                    
                    idModel = ResNetArcFace('IRBlock',
                                            [2,2,2,2],
                                            False).to(x_src.device)
                    idModel.load_state_dict(fix_state_dict(torch.load('/home/topaz/Yunan/Yunan/GFPGAN-1024/model/pretrained_models/arcface_resnet18.pth')))
                    for p in idModel.parameters():
                        p.requires_grad = False
                    idModel.eval()
                    
                    out_gray = gray_resize_for_identity(x_tgt_pred_motion)
                    tgt_gray = gray_resize_for_identity(x_tgt_motion)
                    with torch.no_grad():
                        identity_gt = idModel(tgt_gray)
                        identity_out =   idModel(out_gray)
                    loss_motion = F.l1_loss(identity_out.float(), identity_gt.float(), reduction="mean") * args.lambda_motion
                    # loss_motion = (1 - F.cosine_similarity(identity_out, identity_gt, dim=1).mean()) * args.lambda_motion
                else:
                    loss_motion = torch.zeros(1).to(x_src.device)
                # Reconstruction loss
                if x_tgt_pred is not None:
                    # print(x_tgt_pred.shape, 'pred b=========')
                    x_tgt_pred = x_tgt_pred.permute(0, 2, 1, 3, 4)               
                    x_tgt_pred = x_tgt_pred.view(-1, 3, x_tgt_pred.size(3), x_tgt_pred.size(4))
                    
                    # print(x_tgt_pred.shape, 'pred a=========')

                    # dx, dy = model_gen.module.dx, model_gen.module.dy
                    # spatial_size = 32 if args.num_frames==25 else 16
                    # print(x_tgt.shape, 'tgt b=========')
                    # x_tgt = x_tgt[:, :, :, 
                    #     model_gen.module.x_start * 8 : 8 * (model_gen.module.x_start + model_gen.module.spatial_size), 
                    #     model_gen.module.y_start * 8 : 8 * (model_gen.module.y_start + model_gen.module.spatial_size)
                    #     ]
                    x_tgt = x_tgt.permute(0, 2, 1, 3, 4)
                    x_tgt = x_tgt.view(-1, 3, x_tgt.size(3), x_tgt.size(4))
                    
                    if args.dust:
                        mask = mask.expand_as(x_tgt_pred)  # (B, C, H, W)
                        mask_sum = mask.sum() + 1e-8  # Add epsilon to avoid division by zero
                        loss_masked = F.mse_loss(x_tgt_pred * mask, x_tgt * mask, reduction='sum') / mask_sum
                        inv_mask = 1 - mask
                        inv_mask_sum = inv_mask.sum() + 1e-8
                        loss_non_masked = F.mse_loss(x_tgt_pred * inv_mask, x_tgt * inv_mask, reduction='sum') / inv_mask_sum
                        loss_l2 = (0.90 * loss_masked + 0.10 * loss_non_masked) * args.lambda_l2
                        # print('mask_num: ', mask_sum, 'loss_masked: ', 0.80 *loss_masked, 'inv_mask_num: ', inv_mask_sum, 'loss_non_masked: ', 0.20*loss_non_masked, '--------------------------------')
                        # print(torch.max(mask), torch.min(mask))
                        # print('--------------------------------')
                        # import pdb; pdb.set_trace()
                    else:
                        loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                    loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                else: 
                    loss_l2 = torch.zeros(1).to(x_src.device)
                    loss_lpips = torch.zeros(1).to(x_src.device)

                loss = loss_latent + loss_l2 + loss_lpips + loss_motion
                if args.lambda_g > 0:
                    # KL loss
                    if torch.cuda.device_count() > 1:
                        loss_g, g_fake_pred = model_reg.module.g_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=None, args=args)
                    else:
                        loss_g, g_fake_pred = model_reg.g_loss(latents=latents_pred, prompt_embeds=prompt_embeds, neg_prompt_embeds=None, args=args)
                    loss = loss + loss_g
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

            
                if args.lambda_g > 0:
                    """
                    d loss: let lora model closed to generator 
                    """
                    if torch.cuda.device_count() > 1:
                        loss_d, loss_d_real, loss_d_fake, d_real_pred, d_fake_pred = model_reg.module.d_loss(latents=latents_pred, gt_latent=latents_tgt, prompt_embeds=prompt_embeds, args=args)
                    else:
                        loss_d, loss_d_real, loss_d_fake, d_real_pred, d_fake_pred = model_reg.d_loss(latents=latents_pred, gt_latent=latents_tgt, prompt_embeds=prompt_embeds, args=args)
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(layers_to_opt_reg, args.max_grad_norm)
                    optimizer_reg.step()
                    lr_scheduler_reg.step()
                    optimizer_reg.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    
                    logs = {}
                    logs["epoch"] = epoch
                    # log all the losses
                    if args.lambda_g > 0:
                        logs["loss_d"] = loss_d.detach().item()
                        logs["loss_real"] = loss_d_real.detach().item()
                        logs["loss_fake"] = loss_d_fake.detach().item()
                        logs["d_real_pred"] = d_real_pred.detach().item()
                        logs["d_fake_pred"] = d_fake_pred.detach().item()
                        logs["loss_g"] = loss_g.detach().item()
                        logs["g_fake_pred"] = g_fake_pred.detach().item()
                    if args.dust:
                        logs["loss_masked"] = loss_masked.detach().item()
                        logs["loss_non_masked"] = loss_non_masked.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_latent"] = loss_latent.detach().item()
                    logs["loss_motion"] = loss_motion.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    logs["loss"] = loss.detach().item()
                    logs["lr_vace"] = lr_scheduler_vace.get_last_lr()[0]
                    logs["lr_prompt"] = lr_scheduler_prompt.get_last_lr()[0]
                    logs["lr_decoder"] = lr_scheduler_decoder.get_last_lr()[0]
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_g_{global_step}")
                        accelerator.unwrap_model(model_gen).unet.save_pretrained(outf)

                        outf = os.path.join(args.output_dir, "checkpoints", f"model_d_{global_step}.pth")
                        torch.save(accelerator.unwrap_model(model_reg).unet_update.state_dict(), outf)
                        # accelerator.unwrap_model(model_reg).unet_update.save(outf)
                    accelerator.log(logs, step=global_step)

if __name__ == "__main__":
    args = parse_args()
    main(args)
