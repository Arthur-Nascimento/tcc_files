import cv2
import nibabel as nib
import numpy as np
import os


images_path = "datasets/tomo/img"
masks_path = "datasets/tomo/mask"

os.makedirs('datasets/tomo_img', exist_ok=True)
os.makedirs('datasets/tomo_img/img', exist_ok=True)
os.makedirs('datasets/tomo_img/mask', exist_ok=True)
total = 0

for file in sorted(os.listdir(images_path)):
    img_nib = nib.load(os.path.join(images_path, file)).get_fdata()
    mask_nib = nib.load(os.path.join(masks_path, "segmentation"+file[-(len(file)-6):])).get_fdata()
    for i in range(img_nib.shape[2]):
        #new_img = cv2.imread(img_nib[:,:,i], cv2.IMREAD_GRAYSCALE)
        #new_mask = cv2.imread(mask_nib[:,:,i], cv2.IMREAD_UNCHANGED)
        new_img_name = os.path.join("datasets/tomo_img/img", str(total)+".png")
        new_mask_name = os.path.join("datasets/tomo_img/mask", str(total)+".png")
        cv2.imwrite(new_img_name, img_nib[:,:,i])
        cv2.imwrite(new_mask_name, mask_nib[:,:,i])
        total += 1