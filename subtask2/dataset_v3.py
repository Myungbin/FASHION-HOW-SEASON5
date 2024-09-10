import logging
import os
import random
import cv2
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, default_collate

from augmentation import inference_transform, train_transform
from config import CFG

# cutmix_or_mixup = cutmix_or_mixup(18)


def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


class ClassificationDataset(Dataset):
    def __init__(self, root, root1, df, df1, transform=None, crop=True, use_hsv=False, is_train=True):
        self.dataset = pd.concat([df, df1], ignore_index=True)
        self.transform = transform
        self.crop = crop
        self.use_hsv = use_hsv
        self.is_train = is_train

        self.data_path = []
        for i in range(len(df)):
            tmp = os.path.join(root, df.iloc[i]["image_name"])
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

        if self.use_hsv:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR 이미지를 HSV로 변환

        if self.is_train:
            if self.crop and random.random() < CFG.CROP_PROB:
                image = self.bbox_crop(image, data)
        else:
            if self.crop:
                image = self.bbox_crop(image, data)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = data["Color"]

        return image, label

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
        logging.info(f"Use Origianl Images & Remove Background Images")
        logging.info(f"Image Train Transform: {self.train_transform}\n")
        logging.info(f"Image Validation Taransfrom: {self.inference_transform}")

    def get_train_loader(
        self, root, root1, df, df1, batch_size=4, shuffle=True, crop=True, use_hsv=False, use_sampler=False, mixup=False
    ):
        dataset = ClassificationDataset(
            root, root1, df, df1, transform=self.train_transform, crop=crop, use_hsv=use_hsv, is_train=True
        )
        sampler = self.sampler() if use_sampler else None

        if mixup:
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=4,
                sampler=sampler,
                collate_fn=collate_fn,
            )
        else:
            train_loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4, sampler=sampler
            )
        logging.info("Train Data Info ")
        logging.info("------------------------------------------------------------")
        logging.info(f"Train data path: {CFG.TRAIN_ROOT}")
        logging.info(f"Train data size: {len(train_loader.dataset)}")
        logging.info(f"Crop: {crop}")
        logging.info(f"Use Hsv Transform: {use_hsv}")
        logging.info(f"Use Mixup: {mixup}")
        logging.info(f"Use Sampler: {use_sampler}")
        logging.info("------------------------------------------------------------\n")
        return train_loader

    def get_val_loader(self, root, root1, df, df1, batch_size=4, shuffle=True, crop=False, use_hsv=False):
        dataset = ClassificationDataset(
            root, root1, df, df1, transform=self.inference_transform, crop=crop, use_hsv=use_hsv, is_train=False
        )
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
        logging.info("Validation Data Info ")
        logging.info("------------------------------------------------------------")
        logging.info(f"Validation data path: {CFG.VAL_ROOT}")
        logging.info(f"Validation data size: {len(val_loader.dataset)}")
        logging.info(f"Crop: {crop}")
        logging.info(f"Use HSV: {use_hsv}")
        logging.info("------------------------------------------------------------\n")

        return val_loader

    def sampler(self):
        import pandas as pd
        from config import CFG

        df = pd.read_csv(CFG.TRAIN_DF_PATH)
        df0 = df.copy()
        df = pd.concat([df, df0], ignore_index=True)
        labels = df["Color"].values
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        logging.info(f"Class Weights: {class_weights}")
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return sampler


if __name__ == "__main__":
    ...
