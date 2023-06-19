from archs.unet import unet
from archs.fcn8 import fcn8
from functions import *
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


# Variables
filters = 8
shape = (512,512,1)
epochs = 1
batch = 8

# Loading dataset
lung = False
if lung:
    img_path = "./datasets/lungs/img"
    mask_path = "./datasets/lungs/mask"
    classes = 2
else:
    img_path = "./datasets/tomo_img/img"
    mask_path = "./datasets/tomo_img/mask"
    classes = 3
sample_size = 200

X_train, mask, y_train = load_dataset(img_path, mask_path, classes, sample_size)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state=42)
print(X_train.shape, X_test.shape)
# Train and val



model = unet(filters, classes, shape)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = [dice_loss],
              metrics = [dice_coefficient])
model.summary()
checkpoint = ModelCheckpoint(filepath='model.h5', save_freq = 'epoch')
#model.load_weights("model.h5")
history = model.fit(X_train, y_train, epochs=epochs, batch_size = batch, callbacks = [checkpoint], validation_data=(X_test, y_test))

a = X_test[0:10]
test = model.predict(a)



