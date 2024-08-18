import os
from pathlib import Path
from datetime import datetime
import torch


class Config:
    # Directory
    ROOT_PATH = Path(__file__).parent
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_MODEL_PATH = os.path.join(ROOT_PATH, "check_points", current_time)
    LOG_DIR = SAVE_MODEL_PATH
    TRAIN_ROOT = r"C:\workspace\dataset\FashionHow\subtask1\train2"
    VAL_ROOT = r"C:\workspace\dataset\FashionHow\subtask1\val2"

    TRAIN_DF_PATH = r"C:\workspace\dataset\FashionHow\subtask1\Fashion-How24_sub1_train.csv"
    VAL_DF_PATH = r"C:\workspace\dataset\FashionHow\subtask1\Fashion-How24_sub1_val.csv"
    
    # data
    H = 224
    W = 224

    # train
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = torch.cuda.device_count() * 4
    BATCH_SIZE = 256
    EPOCHS = 300
    SEED = 1103
    SHUFFLE = True
    LEARNING_RATE = 5e-6


CFG = Config()


if __name__ == "__main__":
    print(CFG.ROOT_PATH)
