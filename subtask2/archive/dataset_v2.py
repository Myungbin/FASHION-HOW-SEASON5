import logging
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, default_collate

from augmentation import train_transform_v2, inference_transform_v2, cutmix_or_mixup

cutmix_or_mixup = cutmix_or_mixup(18)


def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


class ClassificationDataset(Dataset):
    def __init__(self, root, df, transform=None, crop=True, use_hsv=False):
        self.root = root
        self.dataset = df
        self.transform = transform
        self.crop = crop
        self.use_hsv = use_hsv

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]

        image_path = data["image_name"]
        image_path = os.path.join(self.root, image_path)

        image = Image.open(image_path).convert("RGB")

        if self.use_hsv:
            image = image.convert("HSV")

        if self.crop:
            image = self.bbox_crop(image, data)

        if self.transform is not None:
            image = self.transform(image)

        label = data["Color"]

        return image, label

    def bbox_crop(self, image, data):
        x_min = data["BBox_xmin"]
        x_max = data["BBox_xmax"]
        y_min = data["BBox_ymin"]
        y_max = data["BBox_ymax"]
        bbox = (x_min, y_min, x_max, y_max)
        image = image.crop(bbox)
        return image


class ClassificationDataLoader:
    def __init__(self):
        self.train_transform = train_transform_v2()
        self.inference_transform = inference_transform_v2()

        logging.info("Dataset Info:")
        logging.info("------------------------------------------------------------")
        logging.info(f"Image Train Transform: {self.train_transform}")
        logging.info(f"Image Validation Taransfrom: {self.inference_transform}")

    def get_train_loader(self, root, df, batch_size=4, shuffle=True, crop=True, use_hsv=False, use_sampler=True,
                         use_collate_fn=False):
        dataset = ClassificationDataset(root, df, transform=self.train_transform, crop=crop, use_hsv=use_hsv)

        sampler = self.sampler() if use_sampler else None
        if use_collate_fn:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4,
                                      sampler=sampler, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4,
                                      sampler=sampler)
        logging.info("Train Data Info ")
        logging.info("------------------------------------------------------------")
        logging.info(f"Crop: {crop}")
        logging.info(f"Use HSV: {use_hsv}")
        logging.info(f"Sampler: {use_sampler}")
        logging.info(f"Collate: {use_collate_fn}")
        return train_loader

    def get_val_loader(self, root, df, batch_size=4, shuffle=True, crop=False, use_hsv=False):
        dataset = ClassificationDataset(root, df, transform=self.inference_transform, crop=crop, use_hsv=use_hsv)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info("Validation Data Info ")
        logging.info("------------------------------------------------------------")
        logging.info(f"Crop: {crop}")
        logging.info(f"Use HSV: {use_hsv}")
        return val_loader

    def sampler(self):
        import pandas as pd
        from config import CFG
        df = pd.read_csv(CFG.TRAIN_DF_PATH)
        labels = df['Color'].values
        class_counts = np.bincount(labels)
        class_weights = 1. / class_counts
        sample_weights = class_weights[labels]
        logging.info(f"Class Weights: {class_weights}")
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler
