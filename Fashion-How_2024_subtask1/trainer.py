import logging
from datetime import datetime

import torch
from tqdm import tqdm

from config import CFG
from log import train_log
from utils import save_model


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, scaler=None, logger=None, patience=20, delta=0.001):
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
            train_log("train", self.model, self.criterion, self.optimizer)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        train_accuracy = 0
        daily_train_accuracy = 0
        gender_train_accuracy = 0
        emb_train_accuracy = 0

        for batch_idx, data in enumerate(tqdm(train_loader)):
            image = data["image"].to(CFG.DEVICE, non_blocking=True)
            daily = data["daily"].to(CFG.DEVICE, non_blocking=True)
            gender = data["gender"].to(CFG.DEVICE, non_blocking=True)
            emb = data["embellishment"].to(CFG.DEVICE, non_blocking=True)

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
            for batch_idx, data in enumerate(tqdm(val_loader)):
                
                image = data["image"].to(CFG.DEVICE, non_blocking=True)
                daily = data["daily"].to(CFG.DEVICE, non_blocking=True)
                gender = data["gender"].to(CFG.DEVICE, non_blocking=True)
                emb = data["embellishment"].to(CFG.DEVICE, non_blocking=True)
                
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

        return validation_loss, validation_accuracy

    def fit(self, train_loader, validation_loader):

        logging.info(f"Train data size: {len(train_loader.dataset)}")
        logging.info(f"Validation data size: {len(validation_loader.dataset)}")

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
                save_model_name = f"{model_name}/{model_name}_{epoch + 1}Epoch.pth"
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
