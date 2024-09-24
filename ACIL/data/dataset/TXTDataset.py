import logging
import os

from PIL import Image
from torch.utils.data import Dataset

logging.getLogger("PIL").setLevel(logging.WARNING)


class TXTDataset(Dataset):
    def __init__(self, root, txt, transform=None, target_transform=None):
        self.root = root
        self.img_path = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform
        with open(os.path.join(root, txt), encoding="utf-8") as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.img_path[index]
        target = self.targets[index]

        with open(path, "rb") as f:
            data = Image.open(f).convert("RGB")

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target
