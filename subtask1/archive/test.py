import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import CFG

from dataset_v2 import ClassificationDataLoader
# from dataset import ClassificationDataLoader
from testloader import TestLoader
from model import EVATiny224
from trainer import Trainer
from utils import seed_everything

if __name__ == "__main__":
    seed_everything(CFG.SEED)
    val_df = pd.read_csv(CFG.VAL_DF_PATH)
    clf = TestLoader()
    val_loader = clf.get_val_loader(CFG.VAL_ROOT, val_df, batch_size=CFG.BATCH_SIZE, shuffle=False, crop=True)
    model = EVATiny224()
    model.load_state_dict(torch.load("C:\workspace\FASHION-HOW\subtask1\model\Best_EVATiny224.pth", map_location="cpu"))
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-7, verbose=True)
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler, True)
    loss, acc = trainer.validation(val_loader)

    print(loss, acc)
