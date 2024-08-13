import logging
import os

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, default_collate

from augmentation import inference_transform, train_transform, cutmix_or_mixup

cutmix_or_mixup = cutmix_or_mixup(18)


def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


def sampler():
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


def bbox_crop(image, data):
    x_min = data["BBox_xmin"]
    x_max = data["BBox_xmax"]
    y_min = data["BBox_ymin"]
    y_max = data["BBox_ymax"]

    image = image[y_min:y_max, x_min:x_max]

    return image


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

        image = cv2.imread(image_path)

        if self.use_hsv:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR 이미지를 HSV로 변환

        if self.crop:
            image = bbox_crop(image, data)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = data["Color"]

        return image, label


class ClassificationDataLoader:
    def __init__(self):
        self.train_transform = train_transform()
        self.inference_transform = inference_transform()

        logging.info("Dataset Info:")
        logging.info("------------------------------------------------------------")
        logging.info(f"Image Train Transform: {self.train_transform}")
        logging.info(f"Image Validation Taransfrom: {self.inference_transform}")

    def get_train_loader(self, root, df, batch_size=4, shuffle=True, crop=True, use_hsv=False, mixup=False):
        dataset = ClassificationDataset(root, df, transform=self.train_transform, crop=crop, use_hsv=use_hsv)
        weight_sampler = sampler()
        if mixup:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4,
                                      sampler=weight_sampler, collate_fn=collate_fn)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4,
                                      sampler=weight_sampler)
        logging.info(train_loader)
        return train_loader

    def get_val_loader(self, root, df, batch_size=4, shuffle=True, crop=False, use_hsv=False):
        dataset = ClassificationDataset(root, df, transform=self.inference_transform, crop=crop, use_hsv=use_hsv)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info(val_loader)
        return val_loader


if __name__ == "__main__":
    ...
