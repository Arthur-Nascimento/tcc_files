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

path = './datasets/tomo_img/mask'
path2 = './datasets/balanced_tomo/mask'


# Saving image as png
'''for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)
    tmp = np.array(np.unique(img, return_counts = True))
    if tmp.shape[1] > 1:
        shutil.copy(os.path.join(path, file), os.path.join(path2[:-4], 'mask',file))
        shutil.copy(os.path.join(path[:-4], 'img', file), os.path.join(path2[:-4], 'img', file))
        print(f"Saved file {file}")'''
while True:
    id = int(input("Numero da imagem: "))
    image = f"{path2}/{id}.png"

    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    '''f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(X_test[image_id], cmap = 'gray')
    axarr[0,1].imshow(np.argmax(y_test[image_id,:,:], axis=-1), cmap = 'gray', vmin =0, vmax = 2)
    axarr[1,0].imshow(X_test[image_id], cmap = 'gray')
    axarr[1,1].imshow(np.argmax(raw[image_id,:,:], axis=-1), cmap = 'gray', vmin =0, vmax = 2)   '''
    plt.imshow(img, cmap = 'gray', vmin=0, vmax=2)
    plt.show()
