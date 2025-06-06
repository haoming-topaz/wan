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
        target_size=(512, 512),
        device_id=0,        
    )

    # points = [[(0, 0), (1, 1)]]
    # labels = [[0, 1]]
    points = [[(1, 1)]]
    labels = [[1]]
    input_points = torch.tensor(points), torch.tensor(labels)
    input_image, input_points = pipe.prepare_input(
        input_image=Image.open('./sample_inputs/cars.jpg'),
        input_points=input_points,
    )
    print(input_image.shape, input_points[0].shape, input_points[1].shape)
    
    output_mask = pipe(input_image, input_points)
    output_dir = 'sample_output'
    output_path = os.path.join(output_dir, 'mask.png')
    output_mask.save(output_path)

    print(f"Frames saved to {output_path}")


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
