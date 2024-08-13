import logging
import os

import cv2
from torch.utils.data import DataLoader, Dataset

from augmentation import inference_transform, train_transform


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
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.crop:
            image = self.bbox_crop(image, data)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

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

    def get_train_loader(self, root, df, batch_size=4, shuffle=True, crop=True):
        dataset = ClassificationDataset(root, df, transform=self.train_transform, crop=crop)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info(f"Train Crop: {crop}")
        return train_loader

    def get_val_loader(self, root, df, batch_size=4, shuffle=True, crop=False):
        dataset = ClassificationDataset(root, df, transform=self.inference_transform, crop=crop)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info(f"Validation Crop: {crop}")
        return val_loader
