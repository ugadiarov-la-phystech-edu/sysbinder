import glob

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image


class DummyDataset(Dataset):
    def __init__(self, img_size=128, img_channels=3, length=100000, *args, **kwargs):
        self.img_size = img_size
        self.img_channels = img_channels
        self.length = length

    def __getitem__(self, idx):
        return torch.zeros(self.img_channels, self.img_size, self.img_size, dtype=torch.float32)

    def __len__(self):
        return self.length


class GlobDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        return tensor_image
