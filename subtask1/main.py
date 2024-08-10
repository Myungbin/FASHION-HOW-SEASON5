import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import CFG
from dataset import ClassificationDataLoader

from model import EVATiny
from trainer import Trainer

from log import set_logging

def main():
    
    set_logging()
    train_root = r"C:\workspace\dataset\FashionHow\subtask1\train/"
    val_root = r"C:\workspace\dataset\FashionHow\subtask1\val"
    train_df = pd.read_csv(r"C:\workspace\dataset\FashionHow\subtask1\Fashion-How24_sub1_train.csv")
    val_df = pd.read_csv(r"C:\workspace\dataset\FashionHow\subtask1\Fashion-How24_sub1_val.csv")
    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(train_root, train_df, batch_size=CFG.BATCH_SIZE, shuffle=True, crop=False)
    val_loader = clf.get_val_loader(val_root, val_df, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=False)
    model = EVATiny()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
