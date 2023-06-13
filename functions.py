import cv2
import os 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def remap_array(element):
    mapping = {0:0, 127:1, 255:2}
    if element in mapping:
        return mapping[element]
    else:
        return element

def load_dataset(img_path, mask_path, classes, sample_size):
    images = []
    masks = []
    img_files = sorted(os.listdir(img_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
    mask_files = sorted(os.listdir(mask_path), key=lambda x: int(''.join(filter(str.isdigit, x))))

    for file in img_files:
        image = cv2.imread(os.path.join(img_path, file), cv2.IMREAD_GRAYSCALE)
        images.append(image)
        if len(images) >= sample_size:
            break

    for file in mask_files:
        if classes == 2:
            # Masks will have 0 or 255 values
            image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_GRAYSCALE)
            _, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
        else:
            # Masks will keep their natural values
            image = cv2.imread(os.path.join(mask_path, file), cv2.IMREAD_UNCHANGED)
        masks.append(image)
        if len(masks) >= sample_size:
            break
    
    masks = np.array(masks)
    # Normalizing images
    images = np.array(images) / 255
    if classes == 2:
        masks_one_hot = np.array(tf.one_hot(masks / 255, classes, axis=-1)).astype(np.float32)
    else:
        masks = np.vectorize(remap_array)(masks).astype(np.float32)
        masks_one_hot = np.array(tf.one_hot(masks, classes, axis=-1)).astype(np.float32)
        masks /= 2.0
    show_data_sample(images[55,:,:], masks[55,:,:], masks_one_hot[55,:,:], 3)
    return images, masks, masks_one_hot


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