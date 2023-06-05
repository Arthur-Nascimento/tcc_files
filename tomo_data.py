import nibabel as nib
import numpy as np
import os

if __name__ == '__main__':
    datasets_path = "./datasets/tomo"
    imgs_path = os.path.join(datasets_path, "img/volume_pt1")
    masks_path = os.path.join(datasets_path, "mask")
    train_amount = 0.8
    volumes_num = 1
    total_volumes = volumes_num*2


    images_path = [f"{imgs_path}/volume-{i}.nii" for i in range(total_volumes)]
    masks_path = [f"{masks_path}/segmentation-{i}.nii" for i in range(total_volumes)]
    imgs = []
    masks = []
    for file in images_path:
        img = nib.load(file).get_fdata()
        for i in range(img.shape[2]):
            imgs.append(img[:,:,i])
    imgs = np.transpose(imgs, axes=[1,2,0])

    for file in masks_path:
        img = nib.load(file).get_fdata()
        for i in range(img.shape[2]):
            masks.append(img[:,:,i])
    masks = np.transpose(masks, axes=[1,2,0])

    print(imgs.shape, masks.shape)

    total_samples = imgs.shape[2]
    toTrain = int(train_amount * total_samples)
    toVal = total_samples - toTrain

    X = imgs[:,:,:toTrain]
    X_val = imgs[:,:,toTrain:]
    y = masks[:,:,:toTrain]
    y_val = masks[:,:,toVal:]



