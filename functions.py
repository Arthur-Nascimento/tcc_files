import cv2
import os 
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


def remap_array(element):
    mapping = {0:0, 127:1, 255:2}
    if element in mapping:
        return mapping[element]
    else:
        return element

def load_dataset(img_path, mask_path, classes, sample_size, shape):
    images = []
    masks = []
    img_files = sorted(os.listdir(img_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
    mask_files = sorted(os.listdir(mask_path), key=lambda x: int(''.join(filter(str.isdigit, x))))

    for file in img_files:
        image = cv2.imread(os.path.join(img_path, file), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, shape)
        images.append(image)
        if len(images) >= sample_size:
            break

    for file in mask_files:
        if classes < 3:
            # Masks will have 0 or 255 values
            image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, shape)
            _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        else:
            # Masks will keep their natural values
            image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, shape)
        masks.append(image)
        if len(masks) >= sample_size:
            break
    
    masks = np.array(masks)
    # Normalizing images
    images = np.array(images) / 255
    if classes > 2:
            masks = np.vectorize(remap_array)(masks).astype(np.float16)
            masks = np.array(tf.one_hot(masks, classes, axis=-1)).astype(np.float16)
    else:
        masks = masks // 255
    return images, masks


def show_data_sample(image, mask, one_hot_mask, classes):
    fig, axes = plt.subplots(nrows=1, ncols=3)

    axes[0].imshow(image, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1)
    axes[1].axis('off')

    hot_mask = np.argmax(one_hot_mask, axis=-1)
    hot_mask = hot_mask / (classes-1)
    axes[2].imshow(hot_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].axis('off')

    axes[0].set_title('Image 1')
    axes[1].set_title('Image 2')
    axes[2].set_title('Image 3')

    plt.show()

# Metrics

def dice_coefficient(y_true, y_pred, smooth=1):
    num_classes = K.int_shape(y_pred)[-1]
    if num_classes > 2:
        dice = 0
        for index in range(num_classes):    
            y_true_class = y_true[..., index]
            y_pred_class = y_pred[..., index]
            y_true_class = K.cast(y_true_class, dtype='float32')  # Convert y_true to float16
            intersection = K.sum(K.abs(y_true_class * y_pred_class), axis=-1)
            union = K.sum(y_true_class, axis=-1) + K.sum(y_pred_class, axis=-1)
            dice += (2.0 * intersection + smooth) / (union + smooth)
        dice /= num_classes
    else:
        y_true = K.cast(y_true, dtype='float32')  # Convert y_true to float32
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coefficient(y_true, y_pred)
    return loss





if __name__ == '__main__':
    lung = True
    if lung:
        img_path = "./datasets/lungs/img"
        mask_path = "./datasets/lungs/mask"
        classes = 2
    else:
        img_path = "./datasets/tomo_img/img"
        mask_path = "./datasets/tomo_img/mask"
        classes = 3
    sample_size = 200

    X, y, y2 = load_dataset(img_path, mask_path, classes, sample_size)
    print(X.shape, y.shape, y2.shape)
    for i in range(20,sample_size):
        show_data_sample(X[i], y[i], y2[i], classes = classes)

def load_dataset_2(img_path, mask_path, classes, sample_size, shape):
    images = []
    masks = []
    img_files = sorted(os.listdir(img_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
    mask_files = sorted(os.listdir(mask_path), key=lambda x: int(''.join(filter(str.isdigit, x))))

    if classes == 2:
        img_files = sorted(os.listdir(img_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
        mask_files = sorted(os.listdir(mask_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
        for file in img_files:
            image = cv2.imread(os.path.join(img_path, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, shape)
            images.append(image)
            if len(images) >= sample_size:
                break
        images = np.array(images) / 255
        for file in mask_files:
            # Masks will have 0 or 255 values
            image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, shape)
            _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
            masks.append(image)
            if len(masks) >= sample_size:
                break     
        masks = np.array(masks)
        # Normalizing images
        masks = masks // 255

    elif classes > 2:
        img_files = sorted(os.listdir(img_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
        mask_files = sorted(os.listdir(mask_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
        for file in img_files:
            image = cv2.imread(os.path.join(img_path, file), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, shape)
            images.append(image)
            if len(images) >= sample_size:
                break
        images = np.array(images) / 255
        for file in mask_files:
            # Masks will keep their natural values
            image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_UNCHANGED)
            image = cv2.resize(image, shape)
        masks = np.vectorize(remap_array)(masks).astype(np.float16)
        masks = np.array(tf.one_hot(masks, classes, axis=-1)).astype(np.float32)
    
    return images, masks