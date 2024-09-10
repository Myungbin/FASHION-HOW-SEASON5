import logging
import os
import random

import cv2
from torch.utils.data import DataLoader, Dataset

from augmentation import inference_transform, train_transform
from config import CFG

CROP_PROB = CFG.CROP_PROB


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
        logging.info(f"Image Train Transform: {self.train_transform}\n")
        logging.info(f"Image Validation Taransfrom: {self.inference_transform}\n")

    def get_train_loader(self, root, df, crop=True):
        dataset = ClassificationDataset(root, df, transform=self.train_transform, crop=crop)
        train_loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=4)

        logging.info("Train Data Info ")
        logging.info("------------------------------------------------------------")
        logging.info(
            f"Train with {len(dataset)} samples. Crop: {crop}, Crop_prob: {CROP_PROB}\nTrain Root: {CFG.TRAIN_ROOT}"
        )
        logging.info("------------------------------------------------------------\n")

        return train_loader

    def get_val_loader(self, root, df, crop=False):
        dataset = ClassificationDataset(root, df, transform=self.inference_transform, crop=crop)
        val_loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=4)
        logging.info("Validation Data Info ")
        logging.info("------------------------------------------------------------")
        logging.info(
            f"Validation with {len(dataset)} samples. Crop: {crop}, Crop_prob: {CROP_PROB}\nValidation Root: {CFG.VAL_ROOT}"
        )
        logging.info("------------------------------------------------------------\n")

        return val_loader
