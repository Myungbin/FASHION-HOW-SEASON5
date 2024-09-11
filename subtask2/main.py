import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import CFG
from dataset import ClassificationDataLoader
from log import set_logging
from model import EVATiny
from trainer import Trainer
from utils import seed_everything


def main():
    set_logging()
    seed_everything(CFG.SEED)
    train_df = pd.read_csv(CFG.TRAIN_DF_PATH)
    val_df = pd.read_csv(CFG.VAL_DF_PATH)
    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(
        CFG.TRAIN_ROOT,
        train_df,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        crop=True,
        use_hsv=False,
        use_sampler=True,
        mixup=False,
    )
    val_loader = clf.get_val_loader(
        CFG.VAL_ROOT, val_df, batch_size=CFG.BATCH_SIZE, shuffle=True, crop=True, use_hsv=False
    )
    model = EVATiny()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
