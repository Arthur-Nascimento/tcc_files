import cv2
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt


images_path = "datasets/tomo/img"
masks_path = "datasets/tomo/mask"

os.makedirs('datasets/tomo_img', exist_ok=True)
os.makedirs('datasets/tomo_img/img', exist_ok=True)
os.makedirs('datasets/tomo_img/mask', exist_ok=True)
total = 0

for file in sorted(os.listdir(images_path)):
    img_nib = nib.load(os.path.join(images_path, file)).get_fdata()
    mask_nib = nib.load(os.path.join(masks_path, "segmentation"+file[-(len(file)-6):])).get_fdata()
    for i in range(mask_nib.shape[2]):
        new_mask = mask_nib[:, :, i] / 2.0
        new_img_name = os.path.join("datasets/tomo_img/img", str(total)+".png")
        new_mask_name = os.path.join("datasets/tomo_img/mask", str(total)+".png")

        new_mask = (new_mask*255).astype(np.uint8)
        cv2.imwrite(new_img_name, np.uint8(img_nib[:,:,i]))
        cv2.imwrite(new_mask_name, new_mask)
        total += 1