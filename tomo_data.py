import nibabel as nib
import numpy as np
import os
import cv2


train_amount = 0.8
tomo = False

if tomo:
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
else:
  if not os.path.isdir("mask"):
    !unzip data.zip

  input_path = os.listdir("img/")
  masks = []
  imgs = []
  norm = 0
  vali = 0
  for file in input_path:
      img_path = os.path.join("img/", file)
      mask_path = os.path.join("mask/", file)
      img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
      mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
      #img = cv2.resize(img, (512, 512))
      #mask = cv2.resize(mask, (512, 512))
      img = np.asarray(img)
      mask = np.asarray(mask)
      imgs.append(img)
      masks.append(mask)


  total_samples = imgs.shape[0]
  toTrain = int(train_amount * total_samples)
  X = imgs[:,:,:toTrain]
  X_val = imgs[:,:,toTrain:]
  y = masks[:,:,:toTrain]
  y_val = masks[:,:,toTrain:]
