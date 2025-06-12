import torch
import json
import random
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class WanSegDataset(Dataset):
    """
    Keep (width, height) for point references.
    """
    def __init__(
            self,
            sources: dict,
            auto_init=True,
            target_size=None,
            downscale_ratio=16,
            enable_batch_sampling=False,
        ):
        super().__init__()
        self.sources = sources
        self.target_size = target_size
        self.target_area = target_size[0] * target_size[1]
        self.downscale_ratio = downscale_ratio
        self.enable_batch_sampling = enable_batch_sampling

        self.image_list = []
        self.mask_list = []
        self.anno_list = []
        self.length = 0
        self.bins = None
        self.idx_to_bin = []

        if auto_init:
            self.initialize()

    def initialize(self):
        for name, filepath in self.sources.items():
            if name == 'coco':
                with open(filepath, 'r') as rf:
                    for line in rf:
                        item = json.loads(line)
                        self.image_list.append(item['image'])
                        self.mask_list.append(item['mask'])
                        self.anno_list.append([(x, 1) for x in item['points']])

        self.length = len(self.image_list)
        if self.enable_batch_sampling:
            self.organize()

    def organize(self):
        bins = dict()
        for idx, image in enumerate(self.image_list):
            image = Image.open(image)
            width, height = image.size
            rescale_ratio = (self.target_area / width / height) ** 0.5
            rescaled_w = int(width * rescale_ratio) // 16
            rescaled_h = int(height * rescale_ratio) // 16

            if (rescaled_w, rescaled_h) not in bins:
                bins[(rescaled_w, rescaled_h)] = []
            bins[(rescaled_w, rescaled_h)].append(idx)
            self.idx_to_bin.append((rescaled_w, rescaled_h))

        self.bins = bins

    def get_batch(self, idx, batch_size):
        assert self.enable_batch_sampling, "Batch sampling is not enabled"
        
        indices = random.choices(self.bins[self.idx_to_bin[idx]], k=batch_size)
        items = [self.__getitem__(idx) for idx in indices]
        images, masks, points, paths = zip(*items)
        images = torch.stack(images)
        masks = torch.stack(masks)
        
        coords = torch.stack([torch.tensor(point[0]) for point in points]).unsqueeze(1)
        labels = torch.stack([torch.tensor(point[1]) for point in points]).unsqueeze(1)
        paths = list(paths)

        return images, masks, (coords, labels), paths
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):        
        image = Image.open(self.image_list[idx])
        mask = Image.open(self.mask_list[idx])
        points = random.choice(self.anno_list[idx])

        rescale_ratio = (self.target_area / image.size[0] / image.size[1]) ** 0.5
        rescaled_w = int(image.size[0] * rescale_ratio) // 16 * 16
        rescaled_h = int(image.size[1] * rescale_ratio) // 16 * 16

        meta = []
        for point, label in [points]:
            meta.append((
                (int(point[0] * rescaled_w / image.width), 
                 int(point[1] * rescaled_h / image.height)), 
                label
            ))

        image = image.convert('RGB').resize((rescaled_w, rescaled_h))
        image = TF.to_tensor(image).sub_(0.5).div_(0.5).unsqueeze(1)

        mask = mask.convert('L').resize((rescaled_w, rescaled_h))
        mask = TF.to_tensor(mask).sub_(0.5).div_(0.5)

        return image, mask, meta[0], (self.image_list[idx], self.mask_list[idx])


def main():
    sample_dataset = WanSegDataset(
        sources={
            'coco': '/home/topaz/haoming/seg/data/coco.jsonl',
        },
        auto_init=True,
        target_size=(128, 128),
        downscale_ratio=16,
    )

    print(sample_dataset.get_batch(0, 10))


if __name__ == "__main__":
    main()
