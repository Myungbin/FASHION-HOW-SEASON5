import logging
import os
import random

import cv2
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from augmentation import inference_transform, train_transform
from config import CFG

CROP_PROB = CFG.CROP_PROB


class ClassificationDataset(Dataset):
    def __init__(self, root0, root1, df0, df1, transform=None, crop=True):
        self.dataset = pd.concat([df0, df1], ignore_index=True)
        self.root = root0 + root1
        self.transform = transform
        self.crop = crop

        self.data_path = []
        for i in range(len(df0)):
            tmp = os.path.join(root0, df0.iloc[i]["image_name"])
            self.data_path.append(tmp)
        for i in range(len(df1)):
            tmp = os.path.join(root1, df1.iloc[i]["image_name"])
            self.data_path.append(tmp)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset.iloc[idx]
        image_path = self.data_path[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.crop and random.random() < CROP_PROB:
            image = self.bbox_crop(image, data)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        daily = data["Daily"]
        gender = data["Gender"]
        embellishment = data["Embellishment"]

        return image, daily, gender, embellishment

    def bbox_crop(self, image, data):
        x_min = data["BBox_xmin"]
        x_max = data["BBox_xmax"]
        y_min = data["BBox_ymin"]
        y_max = data["BBox_ymax"]

        image = image[y_min:y_max, x_min:x_max]

        return image


class ClassificationDataLoader:
    def __init__(self):
        self.train_transform = train_transform()
        self.inference_transform = inference_transform()

        logging.info("Dataset Info:")
        logging.info("------------------------------------------------------------")
        logging.info(f"Image Train Transform: {self.train_transform}")
        logging.info(f"Image Validation Taransfrom: {self.inference_transform}")

    def get_train_loader(self, root, root0, df, df1, batch_size=4, shuffle=True, crop=True):
        dataset = ClassificationDataset(root, root0, df, df1, transform=self.train_transform, crop=crop)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info(f"Train with {len(dataset)} samples. Crop: {crop}, Crop_prob: {CROP_PROB}")

        return train_loader

    def get_val_loader(self, root, root0, df, df1, batch_size=4, shuffle=True, crop=False):
        dataset = ClassificationDataset(root, root0, df, df1, transform=self.train_transform, crop=crop)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info(f"Validation with {len(dataset)} samples. Crop: {crop}, Crop_prob: {CROP_PROB}")

        return val_loader
