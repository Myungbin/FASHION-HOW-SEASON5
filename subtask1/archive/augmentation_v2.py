import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2


def train_transform_v2():
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandAugment(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        ]
    )
    return transforms


def inference_transform_v2():
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(224, 224), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms
