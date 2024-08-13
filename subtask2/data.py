import os.path
import numpy as np

from PIL import Image
from config import CFG
import pandas as pd
import matplotlib.pyplot as plt
def bbox_crop(image, data):
    x_min = data["BBox_xmin"]
    x_max = data["BBox_xmax"]
    y_min = data["BBox_ymin"]
    y_max = data["BBox_ymax"]
    bbox = (x_min, y_min, x_max, y_max)
    image = image.crop(bbox)
    return image

i = 623
dataset = pd.read_csv(CFG.TRAIN_DF_PATH)
a = dataset.iloc[i]['image_name']

image = Image.open(os.path.join(CFG.TRAIN_ROOT, a)).convert("RGB")
image = bbox_crop(image, dataset.iloc[i])

img_np = np.array(image) ## 행렬로 변환된 이미지
plt.imshow(img_np) ## 행렬 이미지를 다시 이미지로 변경해 디스플레이
plt.show() ## 이미지 인터프린터에 출
