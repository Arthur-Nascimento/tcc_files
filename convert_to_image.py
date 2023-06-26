import cv2
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt


images_path = "datasets/tomo/img"
masks_path = "datasets/tomo/mask"

new_mask_path = 'datasets/tomo_test/mask'
new_img_path = 'datasets/tomo_test/img'
liver=(150,30)

os.makedirs(new_img_path, exist_ok=True)
os.makedirs(new_mask_path, exist_ok=True)
total = 0

for file in sorted(os.listdir(images_path)):
    # Image
    img_nib = nib.load(os.path.join(images_path, file)).get_fdata()
    img_nib[img_nib < 30] = 0
    img_nib[img_nib > 150] = 255
    img_nib[img_nib >= 30] = ((img_nib[img_nib >= 30] - 30) // (150 - 30)) * 255
    # Mask
    mask_nib = nib.load(os.path.join(masks_path, "segmentation"+file[-(len(file)-6):])).get_fdata()
    print(mask_nib.shape, img_nib.shape)
    print(np.max(img_nib), np.min(img_nib))
    print(np.unique(img_nib, return_counts=True))
    for i in range(mask_nib.shape[2]):
        new_mask = mask_nib[..., i]
        new_img_name = os.path.join("datasets/tomo_test/img", str(total)+".png")
        new_mask_name = os.path.join("datasets/tomo_test/mask", str(total)+".png")
        cv2.imwrite(new_img_name, np.uint8(img_nib[...,i]))
        cv2.imwrite(new_mask_name, new_mask)
        total += 1