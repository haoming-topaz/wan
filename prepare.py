import torch
import numpy as np
import os
from PIL import Image
from wan.seg import WanSeg, prepare_text_input
from wan.configs import WAN_CONFIGS


def test_pipeline():
    config = WAN_CONFIGS['t2v-1.3B']
    pipe = WanSeg(
        config=config,
        checkpoint_dir='./Wan2.1-VACE-1.3B',
        pretrained_models_dir='./pretrained_models',
        device_id=0,
        rank=0,
    )

    # points = [[(0, 0), (1, 1)]]
    # labels = [[0, 1]]
    points = [[(1, 1)]]
    labels = [[1]]
    input_points = torch.tensor(points), torch.tensor(labels)
    
    video = pipe.generate(
        input_image=Image.open('./sample_inputs/cars.jpg'),
        input_points=input_points,
    )

    output_dir = 'sample_output'
    os.makedirs(output_dir, exist_ok=True)
    video_frames = (
        torch.clamp(video / 2 + 0.5, min=0.0, max=1.0)
        .permute(1, 2, 3, 0) * 255
    ).cpu().numpy().astype(np.uint8)

    for idx, frame in enumerate(video_frames):
        img = Image.fromarray(frame)
        filename = os.path.join(output_dir, f"frame_{idx+1:05d}.png")
        img.save(filename)

    print(f"Frames saved to {output_dir}")


def gen_text_embeddings():
    prompt = 'segmentation mask, high contrast edges, clean background, clear contours'
    negative_prompt= 'blurry, low contrast, occlusion, background clutter, shadow, noise, overlapping parts'
    
    config = WAN_CONFIGS['t2v-1.3B']    
    prepare_text_input(
        config=config,
        checkpoint_dir='./Wan2.1-VACE-1.3B',
        text=prompt,
        save_path='./pretrained_models/segment_prompt.pt',
    )
    prepare_text_input(
        config=config,
        checkpoint_dir='./Wan2.1-VACE-1.3B',
        text=negative_prompt,
        save_path='./pretrained_models/segment_negative_prompt.pt',
    )


if __name__ == '__main__':
    # gen_text_embeddings()
    test_pipeline()
