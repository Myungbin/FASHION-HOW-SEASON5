import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from scheduler import CosineAnnealingWarmUpRestarts
from config import CFG
# from dataset import ClassificationDataLoader
from dataset_v2 import ClassificationDataLoader

from model import ConvNeXt, EVATiny, Mobile, ColorClassifier
from trainer import Trainer
from utils import seed_everything
from log import set_logging


def main():
    seed_everything(CFG.SEED)
    set_logging()
    train_df = pd.read_csv(CFG.TRAIN_DF_PATH)
    val_df = pd.read_csv(CFG.VAL_DF_PATH)
    clf = ClassificationDataLoader()
    train_loader = clf.get_train_loader(CFG.TRAIN_ROOT, train_df, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=False,
                                        use_hsv=False, use_sampler=False, use_collate_fn=False)
    val_loader = clf.get_val_loader(CFG.VAL_ROOT, val_df, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=False,
                                    use_hsv=False)
    model = ColorClassifier()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=CFG.LEARNING_RATE / 10)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
