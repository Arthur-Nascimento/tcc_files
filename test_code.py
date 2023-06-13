import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


zeros = np.zeros((512,512), dtype =float)
ones = np.ones((512,512), dtype =float)
ones2 = np.ones((512,512), dtype =float)
ones /= 2 
image = np.concatenate([zeros,ones], axis = 0)

plt.imshow(image, cmap='gray')
plt.show()
