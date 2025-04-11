import os
import pickle
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class ImageNet32Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        if train:
            self.data_dir = os.path.join(root_dir, "Imagenet32_train")
            self.batch_files = [f"train_data_batch_{i}" for i in range(1, 11)]
        else:
            self.data_dir = os.path.join(root_dir, "Imagenet32_val")
            self.batch_files = ["val_data"]

        self.data = []
        self.targets = []

        for batch_file in self.batch_files:
            file_path = os.path.join(self.data_dir, batch_file)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="bytes")
                self.data.append(entry[b"data"])
                self.targets.extend(entry[b"labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        # Convert to PIL Image
        img = transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)

        return img, target - 1  # ImageNet labels are 1-indexed, convert to 0-indexed
