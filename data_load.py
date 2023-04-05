import numpy as np
from matplotlib import pyplot as plt


x = np.load('unet/images.npy')
y = np.load('unet/masks.npy')
x_val = np.load('unet/images_val.npy')
y_val = np.load('unet/masks_val.npy')


#plt.ion()
for i in range(len(x_val)):
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(x[i])
    axarr[0,1].imshow(y[i,:,:], cmap = 'gray')
    axarr[1,0].imshow(x_val[i])
    axarr[1,1].imshow(y_val[i,:,:], cmap = 'gray')
    plt.show()
