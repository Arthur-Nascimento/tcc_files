import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K


def unet(filters, classes = 2, shape = (512,512,1)):
    input = Input(shape)

    conv1 = Conv2D(filters*2, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(input)
    # Including batch normalization as possible fix to all black prediction
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters*2, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2))(conv1)

    conv2 = Conv2D(filters*4, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters*4, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size = (2,2))(conv2)

    conv3 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size = (2,2))(conv3)

    conv4 = Conv2D(filters*16, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filters*16, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv4)
    conv4 = BatchNormalization()(conv4)
    #drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2))(conv4)

    conv5 = Conv2D(filters*32, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(filters*32, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv5)
    conv5 = BatchNormalization()(conv5)

    convUp = Conv2DTranspose(filters*16, (2,2), strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv5)
    convUp = Concatenate()([convUp, conv4])
    convUp = Conv2D(filters*16, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)
    convUp = Conv2D(filters*16, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)

    convUp = Conv2DTranspose(filters*8, (2,2), strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)
    convUp = Concatenate()([convUp, conv3])
    convUp = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)
    convUp = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)

    convUp = Conv2DTranspose(filters*4, (2,2), strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)
    convUp = Concatenate()([convUp, conv2])
    convUp = Conv2D(filters*4, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)
    convUp = Conv2D(filters*4, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)

    convUp = Conv2DTranspose(filters*2, (2,2), strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)
    convUp = Concatenate()([convUp, conv1])
    convUp = Conv2D(filters*2, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)
    convUp = Conv2D(filters*2, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(convUp)

    if classes == 2:
        output = Conv2D(1, (1,1), activation = 'sigmoid', kernel_initializer = 'he_uniform')(convUp)
    else:
        output = Conv2D((classes), (1,1), activation = 'softmax', kernel_initializer = 'he_uniform')(convUp)
    
    model = Model(inputs = [input], outputs = [output])
    return model