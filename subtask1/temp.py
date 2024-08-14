import random

crop = True
crop_prob = 0.5

for i in range(10):
    if crop and random.random() < crop_prob:
        print("Cropped")
