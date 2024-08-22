import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from timm.loss import AsymmetricLossSingleLabel
from transformers import get_cosine_schedule_with_warmup
from config import CFG

from dataset import ClassificationDataLoader
from log import set_logging
from model import EVATiny224
from trainer import Trainer
from utils import seed_everything



def main():
    seed_everything(CFG.SEED)
    set_logging()
    train_df = pd.read_csv(CFG.TRAIN_DF_PATH)
    val_df = pd.read_csv(CFG.VAL_DF_PATH)
    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(CFG.TRAIN_ROOT, train_df, batch_size=CFG.BATCH_SIZE, shuffle=True, crop=True)
    val_loader = clf.get_val_loader(CFG.VAL_ROOT, val_df, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=True)
    model = EVATiny224()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    # criterion = AsymmetricLossSingleLabel()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    # scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-7, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
