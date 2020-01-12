from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
import numpy as np
from datetime import datetime
import os
import model
from param import img_height,img_width,train_dirs,valid_dirs,\
     qty_train_samples,qty_valid_samples,GPUConf,useGPU

useGPU(True)

#CHANGE THIS
model_name = "model4"
#PARAMETER
logdir = "logs/scalars/" + model_name +"-"+datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callbacks = TensorBoard(log_dir=logdir)
qty_train_samples = qty_train_samples()
qty_valid_samples = qty_valid_samples()


#input epochs
try:
    epochs = int(input("Input epoch [default:50] : "))
except ValueError:
    epochs = 50

batch_size = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dirs,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dirs,
    color_mode='grayscale',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

myModel = load_model(model_name) or model.createModel()
#myModel.summary()
checkpointer = ModelCheckpoint(filepath=model_name+'.h5', monitor='val_loss',verbose=1, save_best_only=True, mode='auto')

print("Start train model")
myModel.fit_generator(
   train_generator,
   steps_per_epoch=qty_train_samples/batch_size,
   epochs=epochs,
   validation_data=valid_generator,
   validation_steps=qty_valid_samples/batch_size,
   verbose=2,
   callbacks=[tensorboard_callbacks,checkpointer]
)

# serialize model to JSON 
# print("Saving model...!")
# model_json = myModel.to_json()
# with open(model_name+".json", "w") as json_file:
#     json_file.write(model_json)
# serialize weights to HDF5
#myModel.save_weights(model_name+".h5")
#print("Saved "+model_name+" to disk drive")