import logging
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ClassificationDataset(Dataset):
    def __init__(self, root, df, transform=None, crop=True):
        self.root = root
        self.dataset = df
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]

        image_path = data["image_name"]
        image_path = os.path.join(self.root, image_path)

        image = Image.open(image_path).convert("RGB")

        if self.crop:
            image = self.bbox_crop(image, data)

        if self.transform is not None:
            image = self.transform(image)

        daily = data["Daily"]
        gender = data["Gender"]
        embellishment = data["Embellishment"]

        result = {}
        result["image"] = image
        result["daily"] = daily
        result["gender"] = gender
        result["embellishment"] = embellishment

        return result

    def bbox_crop(self, image, data):
        x_min = data["BBox_xmin"]
        x_max = data["BBox_xmax"]
        y_min = data["BBox_ymin"]
        y_max = data["BBox_ymax"]
        bbox = (x_min, y_min, x_max, y_max)
        image = image.crop(bbox)
        return image


def inference_transform():
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


class TestLoader:
    def __init__(self):
        self.inference_transform = inference_transform()

    def get_val_loader(self, root, df, batch_size=4, shuffle=True, crop=False):
        dataset = ClassificationDataset(root, df, transform=self.inference_transform, crop=crop)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info(f"Crop: {crop}")
        return val_loader
