import numpy as np
import os
import cv2


input_path = os.listdir("dataset/img/")
output_path = os.listdir("dataset/mask/")

samples = len(input_path)
val_amount = round(0.25*samples)

X =  np.zeros((samples-val_amount, 512, 512, 3), dtype = "uint8")
y =  np.zeros((samples-val_amount, 512, 512), dtype = "bool")
X_val = np.zeros((val_amount, 512, 512, 3), dtype = "uint8")
y_val = np.zeros((val_amount, 512, 512), dtype = "bool")

norm = 0
vali = 0
for i in range(samples):
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


print(X.shape, x_val.shape)
np.save("images.npy", X)
np.save("masks.npy", y)
np.save("images_val.npy", X_val)
np.save("masks_val.npy", y_val)