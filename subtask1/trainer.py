import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CFG
from log import train_log
from utils import save_model
from torchvision.transforms import v2

# cutmix_daily = v2.CutMix(num_classes=6)
# cutmix_gender = v2.CutMix(num_classes=5)
# cutmix_embellishment = v2.CutMix(num_classes=3)


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, scaler=None, logger=None, patience=50, delta=0.001):
        """Trainer 클래스의 생성자.

        Args:
            model (nn.Module): 학습할 모델.
            criterion (nn.Module): 손실 함수.
            optimizer (torch.optim.Optimizer): 최적화 함수.
            scheduler (torch.optim.lr_scheduler._LRScheduler): 학습 스케줄러.
            scaler (torch.cuda.amp.GradScaler, optional): Mixed precision(혼합 정밀도)를 사용하는 경우 필요한 스케일러.
            logger (bool, optional): 로깅 여부 (기본값: False).

        """
        self.model = model.to(CFG.DEVICE)
        self.criterion = criterion.to(CFG.DEVICE)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.best_loss = 99999
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False

        if logger:
            train_log("train", self.model, self.criterion, self.optimizer, self.scheduler)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        daily_train_accuracy = 0
        gender_train_accuracy = 0
        emb_train_accuracy = 0

        for _, (image, daily, gender, emb) in enumerate(tqdm(train_loader)):
            image = image.to(CFG.DEVICE, non_blocking=True)
            daily = daily.to(CFG.DEVICE, non_blocking=True)
            gender = gender.to(CFG.DEVICE, non_blocking=True)
            emb = emb.to(CFG.DEVICE, non_blocking=True)

            # image, daily = cutmix_daily(image, daily)
            # image, gender = cutmix_gender(image, gender)
            # image, emb = cutmix_embellishment(image, emb)

            with torch.cuda.amp.autocast():
                daily_pred, gender_pred, emb_pred = self.model(image)
                daily_loss = self.criterion(daily_pred, daily)
                gender_loss = self.criterion(gender_pred, gender)
                emb_loss = self.criterion(emb_pred, emb)

                loss = daily_loss + gender_loss + emb_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item() / len(train_loader)

            daily_acc = (daily_pred.argmax(dim=1) == daily).float().mean()
            gender_acc = (gender_pred.argmax(dim=1) == gender).float().mean()
            emb_acc = (emb_pred.argmax(dim=1) == emb).float().mean()

            # if Cuxmix or Mixup is used, label is a tuple
            # daily_acc = (daily_pred.argmax(dim=1) == daily.argmax(dim=1)).float().mean()
            # gender_acc = (gender_pred.argmax(dim=1) == gender.argmax(dim=1)).float().mean()
            # emb_acc = (emb_pred.argmax(dim=1) == emb.argmax(dim=1)).float().mean()


            daily_train_accuracy += daily_acc / len(train_loader)
            gender_train_accuracy += gender_acc / len(train_loader)
            emb_train_accuracy += emb_acc / len(train_loader)

        train_accuracy = (daily_train_accuracy + gender_train_accuracy + emb_train_accuracy) / 3

        return train_loss, train_accuracy

    def validation(self, val_loader):
        self.model.eval()

        validation_loss = 0
        validation_accuracy = 0
        daily_validation_accuracy = 0
        gender_validation_accuracy = 0
        emb_validation_accuracy = 0

        with torch.inference_mode():
            for _, (image, daily, gender, emb) in enumerate(tqdm(val_loader)):

                image = image.to(CFG.DEVICE, non_blocking=True)
                daily = daily.to(CFG.DEVICE, non_blocking=True)
                gender = gender.to(CFG.DEVICE, non_blocking=True)
                emb = emb.to(CFG.DEVICE, non_blocking=True)

                daily_pred, gender_pred, emb_pred = self.model(image)
                daily_loss = self.criterion(daily_pred, daily)
                gender_loss = self.criterion(gender_pred, gender)
                emb_loss = self.criterion(emb_pred, emb)

                loss = daily_loss + gender_loss + emb_loss

                validation_loss += loss.item() / len(val_loader)

                daily_acc = (daily_pred.argmax(dim=1) == daily).float().mean()
                gender_acc = (gender_pred.argmax(dim=1) == gender).float().mean()
                emb_acc = (emb_pred.argmax(dim=1) == emb).float().mean()

                daily_validation_accuracy += daily_acc / len(val_loader)
                gender_validation_accuracy += gender_acc / len(val_loader)
                emb_validation_accuracy += emb_acc / len(val_loader)

            validation_accuracy = (daily_validation_accuracy + gender_validation_accuracy + emb_validation_accuracy) / 3

            """어떤 class를 못 맞추는지 확인이 필요"""
            # logging.info(f"daily: {daily_validation_accuracy}")
            # logging.info(f"gender: {gender_validation_accuracy}")
            # logging.info(f"emb: {emb_validation_accuracy}")

        return validation_loss, validation_accuracy

    def fit(self, train_loader, validation_loader):
        for epoch in range(CFG.EPOCHS):
            avg_train_loss, train_accuracy = self.train(train_loader)
            avg_val_loss, val_accuracy = self.validation(validation_loader)

            log_msg = (
                f"Epoch [{epoch + 1}/{CFG.EPOCHS}] "
                f"Training Loss: {avg_train_loss:.4f} "
                f"Training Accuracy: {train_accuracy:.4f} "
                f"Validation Loss: {avg_val_loss:.4f} "
                f"Validation Accuracy: {val_accuracy:.4f} "
            )

            logging.info(log_msg)

            if self.scheduler is not None:
                self.scheduler.step()
            """Early Stop Logic"""
            if avg_val_loss < self.best_loss - self.delta:
                print("Validation loss decreased, saving model")
                self.best_loss = avg_val_loss
                best_model = self.model
                model_name = self.model.__class__.__name__
                save_model_name = f"Best_{model_name}.pth"
                save_model(best_model, save_model_name)
                self.counter = 0
            else:
                self.counter += 1
                print(f"Early Stopping Counter {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    logging.info(f"Early Stopping Counter {self.counter}/{self.patience}")
                    logging.info("Early stopping triggered")
                    self.early_stop = True
                    break

        return best_model


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate the cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Calculate the probability for the target class
        prob = torch.exp(-ce_loss)

        # Calculate the focal loss
        focal_loss = (self.alpha * (1 - prob) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
