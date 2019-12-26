from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras import backend as K
import numpy as np
from keras.preprocessing import image
import os

import model

#PARAMETER
img_width, img_height = 128,128
train_data_dir = os.getcwd()+'/cnn/traindata/train'
valid_data_dir = os.getcwd()+'/cnn/traindata/valid'
qty_train_samples = 180
qty_valid_samples = 30
epochs = 50
batch_size = 10
#qty_class = 4

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_data_dir,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

myModel = model.createModel()
#myModel.summary()

print("Start train Model")
myModel.fit_generator(
   train_generator,
   steps_per_epoch=qty_train_samples/batch_size,
   epochs=epochs,
   validation_data=valid_generator,
   validation_steps=qty_valid_samples/batch_size,
   verbose=2
)

# serialize model to JSON
model_json = myModel.to_json()
with open("cnn/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
myModel.save_weights("cnn/model.h5")
print("Saved model to disk")