import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import CFG
from dataset_v3 import ClassificationDataLoader
from log import set_logging
from model import EVATiny
from trainer import Trainer
from utils import seed_everything


def main():
    TRAIN_ROOT = r"C:\workspace\dataset\FashionHow\subtask2\train2"
    VAL_ROOT = r"C:\workspace\dataset\FashionHow\subtask2\val2"
    TRAIN_ROOT0 = r"C:\workspace\dataset\FashionHow\subtask2\train"
    VAL_ROOT0 = r"C:\workspace\dataset\FashionHow\subtask2\val"
    set_logging()
    seed_everything(CFG.SEED)
    train_df = pd.read_csv(CFG.TRAIN_DF_PATH)
    val_df = pd.read_csv(CFG.VAL_DF_PATH)
    train_df0 = pd.read_csv(CFG.TRAIN_DF_PATH)
    val_df0 = pd.read_csv(CFG.VAL_DF_PATH)
    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(
        TRAIN_ROOT,
        TRAIN_ROOT0,
        train_df,
        train_df0,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        crop=True,
        use_hsv=False,
        use_sampler=True,
        mixup=False,
    )
    val_loader = clf.get_val_loader(
        VAL_ROOT, VAL_ROOT0, val_df, val_df0, batch_size=CFG.BATCH_SIZE, shuffle=True, crop=True, use_hsv=False
    )
    model = EVATiny()
    scaler = torch.cuda.amp.GradScaler()

    class_weight = torch.FloatTensor(
        compute_class_weight("balanced", classes=train_df.Color.sort_values().unique(), y=train_df.Color)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=CFG.LEARNING_RATE / 100)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
