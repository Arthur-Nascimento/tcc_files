import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from functions import *
import shutil

'''
img_path = "./datasets/tomo_img/img"
mask_path = "./datasets/tomo_img/mask"

filters = 8
shape = (256,256,1)
epochs = 100
batch = 8
lung = False
sample_size = 2000
classes = 3

X_train, y_train = load_dataset(img_path, mask_path, classes, sample_size, (shape[0], shape[1]))
y_train = np.argmax(y_train, axis=-1)    
value, counts = np.unique(y_train, return_counts=True)

for value, count in zip(value, counts):
    print(f"{value}: {count}")
'''

path = './datasets/tomo_img/img'
path2 = './datasets/balanced_tomo/mask'

for file in os.listdir(path2):
    shutil.copy(os.path.join(path, file), os.path.join(path2[:-4], 'img',file))
