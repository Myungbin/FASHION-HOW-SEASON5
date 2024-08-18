import os
from pathlib import Path
from datetime import datetime
import torch


class Config:
    # Directory
    ROOT_PATH: str = Path(__file__).parent
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_MODEL_PATH: str = os.path.join(ROOT_PATH, "check_points", current_time)
    LOG_DIR: str = SAVE_MODEL_PATH

    TRAIN_ROOT: str = r"C:\workspace\dataset\FashionHow\subtask2\train"
    VAL_ROOT: str = r"C:\workspace\dataset\FashionHow\subtask2\val"

    TRAIN_DF_PATH: str = r"C:\workspace\dataset\FashionHow\subtask2\Fashion-How24_sub2_train.csv"
    VAL_DF_PATH: str = r"C:\workspace\dataset\FashionHow\subtask2\Fashion-How24_sub2_val.csv"

    # data
    H: int = 224
    W: int = 224

    # train
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS: int = torch.cuda.device_count() * 4
    BATCH_SIZE: int = 128
    EPOCHS: int = 300
    SEED: int = 1103
    SHUFFLE: bool = True
    LEARNING_RATE: float = 1e-4


CFG = Config()


if __name__ == "__main__":
    print(CFG.ROOT_PATH)
