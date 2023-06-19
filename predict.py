from archs.unet import unet
from archs.fcn8 import fcn8
from functions import *
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# Variables
filters = 8
shape = (512,512,1)

model = unet(filters, 3, shape)
