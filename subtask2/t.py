import pandas as pd
from config import CFG

df = pd.read_csv(CFG.TRAIN_DF_PATH)
labels = df["Color"].values

from sklearn.utils.class_weight import compute_class_weight

class_weight = compute_class_weight('balanced', classes=df.Color.sort_values().unique(), y=df.Color)

print(class_weight)

print(0.01538462/0.00249377)

print(9.45128205/1.53200333)