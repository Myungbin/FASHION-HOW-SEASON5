import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2

# from torchvision.transforms import v2

from config import CFG


def train_transform():
    transform = A.Compose(
        [
            A.OneOf([A.Resize(CFG.H, CFG.W, interpolation=cv2.INTER_CUBIC)], p=1.0),
            # A.VerticalFlip(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.OneOf(
            #     [
            #         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
            #         A.CLAHE(clip_limit=6.0, tile_grid_size=(16, 16)),
            #         A.RandomGamma(gamma_limit=(80, 120)),
            #     ],
            #     p=0.5,
            # ),
            # A.OneOf(
            #     [
            #         A.OpticalDistortion(distort_limit=0.5),
            #         A.GridDistortion(num_steps=5, distort_limit=0.5),
            #     ],
            #     p=0.3,
            # ),
            # A.OneOf(
            #     [
            #         A.MotionBlur(blur_limit=3),
            #         A.MedianBlur(blur_limit=3),
            #         A.GaussianBlur(blur_limit=3),
            #         A.GaussNoise(var_limit=(3.0, 9.0)),
            #     ],
            #     p=0.3,
            # ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20,
                border_mode=4,
                p=0.3,
            ),
            A.CoarseDropout(
                min_holes=1,
                max_holes=6,
                min_width=4,
                min_height=4,
                max_height=int(CFG.H * 0.1),
                max_width=int(CFG.W * 0.1),
                p=0.5,
            ),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    return transform


def inference_transform():
    transform = A.Compose(
        [
            A.Resize(CFG.H, CFG.W, interpolation=cv2.INTER_CUBIC),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    return transform


# def train_transform_v2():
#     transforms = v2.Compose(
#         [
#             v2.ToImage(),
#             v2.Resize(size=(224, 224), antialias=True),
#             v2.ToDtype(torch.uint8, scale=True),
#             v2.RandAugment(),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
#         ]
#     )
#     return transforms


# def inference_transform_v2():
#     transforms = v2.Compose(
#         [
#             v2.ToImage(),
#             v2.Resize(size=(224, 224), antialias=True),
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ]
#     )
#     return transforms


# def cutmix_or_mixup(NUM_CLASSES=18):
#     cutmix = v2.CutMix(num_classes=NUM_CLASSES)
#     mixup = v2.MixUp(num_classes=NUM_CLASSES)
#     cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
#     return cutmix_or_mixup
