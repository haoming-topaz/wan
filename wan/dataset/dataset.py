import torch
import json
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator


class WanSegDataset(Dataset):
    def __init__(self, sources: dict, auto_init=True):
        super().__init__()
        self.sources = sources

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
        point = random.choice(self.anno_list[idx])

        return image, mask, point


def main():
    pass


if __name__ == "__main__":
    main()
