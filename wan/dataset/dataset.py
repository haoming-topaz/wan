import torch
import json
import random
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class WanSegDataset(Dataset):
    def __init__(
            self,
            sources: dict,
            auto_init=True,
            target_size=None,
        ):
        super().__init__()
        self.sources = sources
        self.target_size = target_size
        self.target_area = target_size[0] * target_size[1]

        self.image_list = []
        self.mask_list = []
        self.anno_list = []
        self.length = 0

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
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):        
        image = Image.open(self.image_list[idx])
        mask = Image.open(self.mask_list[idx])
        points = random.choice(self.anno_list[idx])

        rescale_ratio = (self.target_area / image.size[0] / image.size[1]) ** 0.5
        rescaled_w = int(image.size[0] * rescale_ratio) // 16 * 16
        rescaled_h = int(image.size[1] * rescale_ratio) // 16 * 16
        image = image.convert('RGB').resize((rescaled_w, rescaled_h))
        image = TF.to_tensor(image).sub_(0.5).div_(0.5).unsqueeze(1)

        mask = mask.convert('L').resize((rescaled_w, rescaled_h))
        mask_arr = np.array(mask)
        mask = torch.zeros(*mask_arr.shape, dtype=torch.float)
        mask[mask_arr > 128] = 1
        mask = mask.unsqueeze(0)

        points, labels = points
        points = [(int(x[0] * rescale_ratio), int(x[1] * rescale_ratio)) for x in points]
        
        return image, mask, (points, labels)


def main():
    pass


if __name__ == "__main__":
    main()
