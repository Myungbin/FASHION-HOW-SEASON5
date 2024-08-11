import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import CFG
from dataset import ClassificationDataLoader

from model import ConvNeXt
from trainer import Trainer

from log import set_logging

def main():
    set_logging()
    train_df = pd.read_csv(CFG.TRAIN_DF_PATH)
    val_df = pd.read_csv(CFG.VAL_DF_PATH)
    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(CFG.TRAIN_ROOT, train_df, batch_size=CFG.BATCH_SIZE, shuffle=True, crop=True)
    val_loader = clf.get_val_loader(CFG.VAL_ROOT, val_df, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=False)
    model = ConvNeXt()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
