import logging
from datetime import datetime

import torch
from tqdm import tqdm

from config import CFG
from log import train_log
from utils import save_model


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

        for _, data in enumerate(tqdm(train_loader)):
            image = data["image"].to(CFG.DEVICE, non_blocking=True)
            label = data["color"].to(CFG.DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast():
                prediction = self.model(image)
                loss = self.criterion(prediction, label)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item() / len(train_loader)
            acc = (prediction.argmax(dim=1) == label).float().mean()
            train_accuracy += acc / len(train_loader)

        return train_loss, train_accuracy

    def validation(self, val_loader):
        self.model.eval()

        validation_loss = 0
        validation_accuracy = 0

        with torch.inference_mode():
            for _, data in enumerate(tqdm(val_loader)):
                image = data["image"].to(CFG.DEVICE, non_blocking=True)
                label = data["color"].to(CFG.DEVICE, non_blocking=True)

                prediction = self.model(image)
                loss = self.criterion(prediction, label)

                validation_loss += loss.item() / len(val_loader)
                acc = (prediction.argmax(dim=1) == label).float().mean()
                validation_accuracy += acc / len(val_loader)

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
                save_model_name = f"{model_name}_{epoch + 1}Epoch.pth"
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
