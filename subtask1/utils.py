import os
import random

import numpy as np
import torch

from config import CFG


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save_model(model, filename):
    save_path = os.path.join(CFG.SAVE_MODEL_PATH, filename)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def load_model_state_dict(model, file_name):
    path = os.path.join(CFG.SAVE_MODEL_PATH, f"{file_name}")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
