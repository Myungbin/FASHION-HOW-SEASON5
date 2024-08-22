import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import CFG
from dataset_v3 import ClassificationDataLoader
from log import set_logging
from model import EVATiny224
from trainer import Trainer
from utils import seed_everything


def main():
    seed_everything(CFG.SEED)
    set_logging()
    TRAIN_DF_PATH = r"C:\workspace\dataset\FashionHow\subtask1\Fashion-How24_sub1_train.csv"
    VAL_DF_PATH = r"C:\workspace\dataset\FashionHow\subtask1\Fashion-How24_sub1_val.csv"
    TRAIN_ROOT = r"C:\workspace\dataset\FashionHow\subtask1\train2"
    VAL_ROOT = r"C:\workspace\dataset\FashionHow\subtask1\val2"
    TRAIN_ROOT0 = r"C:\workspace\dataset\FashionHow\subtask1\train"
    VAL_ROOT0 = r"C:\workspace\dataset\FashionHow\subtask1\val"

    train_df = pd.read_csv(TRAIN_DF_PATH)
    val_df = pd.read_csv(VAL_DF_PATH)
    train_df0 = pd.read_csv(TRAIN_DF_PATH)
    val_df0 = pd.read_csv(VAL_DF_PATH)

    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(
        TRAIN_ROOT0, TRAIN_ROOT, train_df, train_df0, batch_size=CFG.BATCH_SIZE, shuffle=True, crop=True
    )
    val_loader = clf.get_val_loader(
        VAL_ROOT0, VAL_ROOT, val_df, val_df0, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=True
    )
    model = EVATiny224()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    # criterion = AsymmetricLossSingleLabel()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    # scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-7, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
