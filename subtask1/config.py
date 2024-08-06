import os
from pathlib import Path

import torch


class Config:
    # Directory
    ROOT_PATH = Path(__file__).parent
    SAVE_MODEL_PATH = os.path.join(ROOT_PATH, "check_points")
    LOG_DIR = "logs"

    # data
    H = 224
    W = 224

    # train
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = torch.cuda.device_count() * 4
    BATCH_SIZE = 256
    EPOCHS = 300
    SEED = 42
    SHUFFLE = True
    LEARNING_RATE = 3e-4
    
CFG = Config()


if __name__ == "__main__":
    print(CFG.ROOT_PATH)