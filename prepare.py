import torch
import numpy as np
import os
from PIL import Image
from wan.text2video import WanT2V
from wan.seg import WanSeg, prepare_text_input
from wan.configs import WAN_CONFIGS
import torchvision.transforms.functional as TF


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
    image_path = './sample_inputs/cars.jpg'
    points = [[(5, 8)]]
    labels = [[1]]
    input_points = torch.tensor(points), torch.tensor(labels)
    input_image, input_points = pipe.prepare_input(
        input_image=Image.open(image_path),
        input_points=input_points,
    )
    print(Image.open(image_path).size, input_points[0].shape, input_points[1].shape)
    
    output_mask = pipe(input_image, input_points)
    output_dir = 'sample_outputs'
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


def test_vae():
    config = WAN_CONFIGS['t2v-1.3B']
    pipe = WanT2V(
        config=config,
        checkpoint_dir='./Wan2.1-T2V-1.3B',
        device_id=0,
        rank=0,
    )    
    image = Image.open('/home/topaz/haoming/seg/exp/25-06-07-19-55/eval/step_2/gt_mask_2.png').convert('RGB')
    image = TF.to_tensor(image).sub_(0.5).div_(0.5).to(torch.device('cuda:0')).unsqueeze(1).unsqueeze(0)
    latents = pipe.vae.encode(image)
    outputs = pipe.vae.decode(latents)
    print(len(outputs))
    output = (outputs[0][:, 0, :, :] + 1 * 0.5).clamp(0, 1)
    output = output.permute(1, 2, 0).detach().cpu().numpy()
    output = (output * 255).astype(np.uint8)
    output = Image.fromarray(output)
    output.save('sample_outputs/vae_output.png')


if __name__ == '__main__':
    # gen_text_embeddings()
    test_pipeline()
    # test_vae()
