import os
import json
from PIL import Image, ImageDraw
import numpy as np
import scipy
from tqdm import tqdm
from pycocotools import mask as maskUtils


category_id_to_label = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}


def generate_masks():
    """
    Generate single-instance masks based on annotations.
    """
    
    json_file = "/nfs/indian/haoming/coco/annotations/instances_train2017.json"
    image_dir = "/nfs/indian/haoming/coco/images"
    output_dir = "/nfs/indian/haoming/coco/single_masks"
    os.makedirs(output_dir, exist_ok=True)
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    for img_id, file_name in tqdm(list(id_to_filename.items())):
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
        width, height = None, None
        
        for ann in annotations:
            segm = ann['segmentation']
            instance_id = ann['id']
            category_id = ann['category_id']
            
            if width is None or height is None:
                image_path = os.path.join(image_dir, file_name)
                with Image.open(image_path) as img:
                    width, height = img.size
                    # img.save(f'{img_id}.jpg')

            mask = np.zeros((height, width), dtype=np.uint8)
            if isinstance(segm, list):
                img_mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(img_mask)
                for polygon in segm:
                    xy = [tuple(polygon[i:i+2]) for i in range(0, len(polygon), 2)]
                    draw.polygon(xy, outline=1, fill=1)
                mask = np.array(img_mask)
            elif isinstance(segm, dict):
                rle = maskUtils.frPyObjects(segm, height, width)
                rle = maskUtils.merge(rle) if isinstance(rle, list) else rle
                mask = maskUtils.decode(rle)
            else:
                print(f"Unknown segmentation format for instance {instance_id}")
                continue

            mask_np = (mask * 255).astype(np.uint8)
            mask_filename = f"{img_id}_{instance_id}_{int(ann['area'])}_{category_id_to_label[category_id]}.png"
            mask_output_path = os.path.join(output_dir, mask_filename)
            Image.fromarray(mask_np).save(mask_output_path)


def convert():
    """
    Create a new json file with necessary training information.
    """
    json_file = "/nfs/indian/haoming/coco/annotations/instances_train2017.json"
    image_dir = "/nfs/indian/haoming/coco/images"
    mask_dir = "/nfs/indian/haoming/coco/single_masks"
    with open(json_file, 'r') as rf:
        coco_data = json.load(rf)
    
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    with open("coco.jsonl", "w") as wf:
        count_total, count_written = 0, 0
        for img_id, file_name in tqdm(list(id_to_filename.items())):
            annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
            
            for ann in annotations:
                count_total += 1
                if ann['area'] < 100:
                    continue
                instance_id = ann['id']
                category_id = ann['category_id']

                mask_filename = f"{img_id}_{instance_id}_{int(ann['area'])}_{category_id_to_label[category_id]}.png"
                mask_full_path = os.path.join(mask_dir, mask_filename)
                mask = np.array(Image.open(mask_full_path))
                
                coords = np.argwhere(mask > 0)  # shape: (N, 2), format: (y, x)
                if len(coords) < 5 or len(coords) * 1000 < mask.shape[0] * mask.shape[1]:
                    continue

                com_y, com_x = scipy.ndimage.center_of_mass(mask)
                com = (int(round(com_y)), int(round(com_x)))
                include_center = mask[com] > 0 if 0 <= com[0] < mask.shape[0] and 0 <= com[1] < mask.shape[1] else False

                np.random.shuffle(coords)
                selected_coords = coords[:5] if not include_center else coords[:4].tolist()
                if include_center:
                    selected_coords.append([com[0], com[1]])

                selected_coords = [(int(x), int(y)) for y, x in selected_coords]

                entry = {
                    'image': os.path.join(image_dir, file_name),
                    'mask': mask_full_path,
                    'points': selected_coords,
                }
                
                count_written += 1
                wf.write(json.dumps(entry) + "\n")

    print(f'{count_written}/{count_total} entries dumped.')


if __name__ == '__main__':
    convert()
