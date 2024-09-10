import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from config import CFG


def train_transform():
    transform = A.Compose(
        [
            A.Resize(CFG.H, CFG.W, interpolation=cv2.INTER_CUBIC),
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
