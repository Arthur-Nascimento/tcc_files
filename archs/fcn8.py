import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Add, Conv2DTranspose, Cropping2D
from tensorflow.keras.models import Model

def fcn8(filters, shape = (512,512,1)):
    input = Input(shape)
    #VGG-16 Encoder
    conv1 = Conv2D(filters, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(input)
    conv1 = Conv2D(filters, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv1)
    pool1 = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(conv1)

    conv2 = Conv2D(filters*2, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool1)
    conv2 = Conv2D(filters*2, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv2)
    pool2 = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(conv2)

    conv3 = Conv2D(filters*4, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool2)
    conv3 = Conv2D(filters*4, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv3)
    conv3 = Conv2D(256, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv3)
    pool3 = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(conv3)

    conv4 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool3)
    conv4 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv4)
    conv4 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv4)
    pool4 = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(conv4)

    conv5 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool4)
    conv5 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv5)
    conv5 = Conv2D(filters*8, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv5)
    pool5 = MaxPooling2D((2,2), strides = (2,2), padding = 'same')(conv5)

    conv6 = Conv2D(filters*64, (7,7), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(pool5)
    conv6 = Conv2D(filters*64, (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv6)
    conv6 = Conv2D(2, (1,1), padding = 'same', activation = 'relu', kernel_initializer = 'he_uniform')(conv6)

    conv7 = Conv2DTranspose(2, kernel_size=(4,4), strides=(2,2), padding='valid')(conv6)
    conv7 = Cropping2D((1,1))(conv7)

    add01 = Conv2D(2, (1,1), activation = "relu", padding = 'same')(pool4)

    out1 = Add()([conv7, add01])
    out2 = Conv2DTranspose(2, kernel_size=(4,4), strides=(2,2), padding='valid')(out1)
    out2 = Cropping2D((1,1))(out2)
    add01 = Conv2D(2, (1,1), activation = "relu", padding = 'same')(pool3)
    out2 = Add()([out2, add01])
    output = Conv2DTranspose(1, kernel_size=(16,16),strides = (8,8), padding='valid',  activation = 'sigmoid')(out2)
    output = Cropping2D(cropping=((0,8),(0,8)))(output)
    model = Model(inputs = [input], outputs = [output])

    return model