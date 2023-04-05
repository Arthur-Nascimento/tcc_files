import numpy as np
import os
import cv2


input_path = os.listdir("dataset/img/")
output_path = os.listdir("dataset/mask/")

val_amount = 20
X =  np.zeros((len(input_path)-val_amount, 512, 512, 3), dtype = "uint8")
y =  np.zeros((len(output_path)-val_amount, 512, 512), dtype = "bool")
X_val = np.zeros((val_amount, 512, 512, 3), dtype = "uint8")
y_val = np.zeros((val_amount, 512, 512), dtype = "bool")

norm = 0
vali = 0
for i in range(len(input_path)):
    img_path = os.path.join("dataset/img/", input_path[i])
    mask_path = os.path.join("dataset/mask/", input_path[i])
    img = cv2.imread(img_path)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (512, 512))
    mask = cv2.resize(mask, (512, 512))
    
    if i >= len(input_path)-val_amount:
        X_val[vali, :, :, :] = img
        y_val[vali, :, :] = mask
        vali += 1
    else:
        X[norm, :, :, :] = img
        y[norm, :, :] = mask
        norm += 1

np.save("images.npy", X)
np.save("masks.npy", y)
np.save("images_val.npy", X_val)
np.save("masks_val.npy", y_val)

'''
with open("images.npy", 'w+') as f:
    np.save(f, X)

with open("masks.npy", 'w+') as f:
    np.save(f, y)

with open("images_val.npy", 'w+') as f:
    np.save(f, X_val)

with open("masks_val.npy", 'w+') as f:
    np.save(f, y_val)
'''
