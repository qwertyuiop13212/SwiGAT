import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PairedTransform:
    def __init__(self, image_size=224, training=False):
        self.image_size = image_size
        self.training = training

    def __call__(self, image, gaze):
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        gaze = TF.resize(gaze, [self.image_size, self.image_size], InterpolationMode.BILINEAR)
        if self.training:
            if random.random() < 0.5:
                image = TF.hflip(image)
                gaze = TF.hflip(gaze)
            angle = random.uniform(-5.0, 5.0)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            gaze = TF.rotate(gaze, angle, interpolation=InterpolationMode.BILINEAR)
            image = TF.adjust_brightness(image, random.uniform(0.9, 1.1))
            image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
        image = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        gaze = TF.to_tensor(gaze)
        gaze_min, gaze_max = gaze.amin(), gaze.amax()
        gaze = (gaze - gaze_min) / (gaze_max - gaze_min + 1e-6)
        return image, gaze


class ChestXrayGazeDataset(Dataset):
    REQUIRED_COLUMNS = {"image", "gaze", "label"}

    def __init__(self, csv_path, image_size=224, training=False):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {self.csv_path}. See data_examples/ for the required format."
            )
        self.frame = pd.read_csv(self.csv_path)
        missing = self.REQUIRED_COLUMNS.difference(self.frame.columns)
        if missing:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing)}")
        self.root = self.csv_path.parent
        self.transform = PairedTransform(image_size, training)

    def __len__(self):
        return len(self.frame)

    def _resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = self._resolve(row["image"])
        gaze_path = self._resolve(row["gaze"])
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        with Image.open(gaze_path) as handle:
            gaze = handle.convert("L")
        image, gaze = self.transform(image, gaze)
        return image, gaze, torch.tensor(int(row["label"]), dtype=torch.long)


def build_loaders(config):
    data_cfg = config["data"]
    train_set = ChestXrayGazeDataset(data_cfg["train_csv"], data_cfg["image_size"], True)
    test_set = ChestXrayGazeDataset(data_cfg["test_csv"], data_cfg["image_size"], False)
    common = {
        "batch_size": config["training"]["batch_size"],
        "num_workers": data_cfg.get("num_workers", 4),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)
    return train_loader, test_loader

