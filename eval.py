import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from wan.utils import FlowUniPCMultistepScheduler
from wan.seg import WanSeg
from wan.configs import WAN_CONFIGS
from wan.dataset import WanSegDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='/home/topaz/haoming/seg/exp-wan/25-06-09-13-31/checkpoints/model_20002.pth')
    parser.add_argument("--output_dir", type=str, default="sample_outputs")
    parser.add_argument("--mode", type=str, default="multi", choices=["single", "multi"])
    parser.add_argument("--num_infer_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_size", type=tuple, default=(512, 512))
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    arch_config = WAN_CONFIGS['t2v-1.3B']
    pipe = WanSeg(
        config=arch_config,
        checkpoint_dir='./Wan2.1-VACE-1.3B',
        pretrained_models_dir='./pretrained_models',
        target_size=args.target_size,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipe.load_state_dict(state_dict)
    pipe.to(device)
    pipe.eval()

    # Setup scheduler
    dataset = WanSegDataset(sources={'coco': './data/coco.jsonl'}, auto_init=True, target_size=args.target_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)

    for idx, batch in enumerate(tqdm(dataloader)):
        image, gt_mask, points, paths = batch
        image = image.to(device)[0]
        gt_mask = gt_mask.to(device)

        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            # solver_order=1,
            use_dynamic_shifting=False,
        )
        if args.mode == "single":
            scheduler.set_timesteps(1000, device=device, shift=1)
        else:
            scheduler.set_timesteps(args.num_infer_steps, device=device, shift=5.0)
            
        with torch.autocast(device_type='cuda', dtype=torch.float), torch.no_grad():
            points, labels = torch.stack(points[0]).unsqueeze(0).to(device), points[1].unsqueeze(0).to(device)
            points = points.transpose(-1, -2)

            mask_input = gt_mask.expand(3, -1, -1, -1)
            x0 = pipe.vae.encode([mask_input])[0].detach()

            noise = torch.randn_like(x0)

            if args.mode == "single":
                # Single step flow matching (train style)
                step_idx = np.random.randint(0, scheduler.num_train_timesteps)
                sigma_t = scheduler.sigmas[step_idx]
                timestep = torch.tensor([scheduler.timesteps[step_idx]], device=device)

                noisy_latents = scheduler.add_noise(x0, noise, timestep)
                p0 = pipe.location_encoder(points=(points, labels), boxes=None, masks=None)[0]
                m0 = pipe.vae.encode([image])[0].detach()

                seq_len = 1280
                noise_pred = pipe.model([noisy_latents], t=timestep, vace_context=[p0, m0],
                                        vace_context_scale=1.0, context=pipe.context, seq_len=seq_len)[0]
                latent = (noisy_latents - noise_pred * sigma_t) / (1 - sigma_t)
            else:
                # Multi-step flow matching (inference style)
                latents = torch.randn_like(x0)
                p0 = pipe.location_encoder(points=(points, labels), boxes=None, masks=None)[0]
                m0 = pipe.vae.encode([image])[0].detach()
                seq_len = 1280

                norms = []
                for t in tqdm(scheduler.timesteps, desc="sampling"):
                    timestep = torch.tensor([t], device=device)
                    noise_pred = pipe.model([latents], t=timestep, vace_context=[p0, m0],
                                            vace_context_scale=1.0, context=pipe.context, seq_len=seq_len)[0]
                    latents = scheduler.step(noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False)[0]
                    latents = latents.squeeze(0)

                    latent_norm = latents.norm().item()
                    norms.append(latent_norm)

                latent = latents
                print(f"Latent norm: {norms}")

            # --> **Always decode flow matched latent**
            pred_mask = pipe.vae.model.decoder(latent.unsqueeze(0))[0][0][0]
            pred_mask = torch.where(pred_mask > 0, torch.ones_like(pred_mask, dtype=torch.uint8),
                                    torch.zeros_like(pred_mask, dtype=torch.uint8))
            pred_mask = pred_mask.cpu().numpy() * 255
            pred_mask = Image.fromarray(pred_mask.astype(np.uint8)).convert('RGB')

            pred_mask.save(os.path.join(args.output_dir, f"pred_{idx}.png"))
            Image.open(paths[0][0]).save(os.path.join(args.output_dir, f"input_{idx}.png"))
            Image.open(paths[1][0]).save(os.path.join(args.output_dir, f"gt_mask_{idx}.png"))       
            
            if idx >= 5:
                break


def show_weights():
    pass


if __name__ == "__main__":
    main()
    # show_weights()
