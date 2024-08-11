import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import CFG
from dataset import ClassificationDataLoader
from log import set_logging
from model import EVATiny336
from trainer import Trainer
from utils import seed_everything


def main():
    seed_everything(CFG.SEED)
    set_logging()
    train_df = pd.read_csv(CFG.TRAIN_DF_PATH)
    val_df = pd.read_csv(CFG.VAL_DF_PATH)
    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(CFG.TRAIN_ROOT, train_df, batch_size=CFG.BATCH_SIZE, shuffle=True, crop=True)
    val_loader = clf.get_val_loader(CFG.VAL_ROOT, val_df, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=False)
    model = EVATiny336()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=5e-7, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
    # TODO: eva02_tiny_patch14_336 실험 해보기
