from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
import numpy as np
from datetime import datetime
import os
import model
from param import Config, useGPU, count_files

useGPU(False)
cfg = Config()
#CHANGE THIS
modelName = "model5"
#PARAMETER
logdir = cfg.dirs.log_path + modelName +"-"+datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callbacks = TensorBoard(log_dir=logdir)
qty_train_samples = count_files(cfg.dirs.train_path)
qty_valid_samples = count_files(cfg.dirs.validation_path)


#input epochs
try:
    epochs = int(input("Input epoch [default:50] : "))
except ValueError:
    epochs = 50

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    cfg.dirs.train_path,
    color_mode='grayscale',
    target_size=(cfg.img.width, cfg.img.height),
    batch_size=cfg.model.batchSize,
    class_mode='categorical'
)

valid_generator = valid_datagen.flow_from_directory(
    cfg.dirs.validation_path,
    color_mode='grayscale',
    target_size=(cfg.img.width, cfg.img.height),
    batch_size=cfg.model.batchSize,
    class_mode='categorical'
)

try:
    myModel = load_model(modelName+'.h5')
except:
    myModel = model.createModel()
#myModel.summary()
checkpointer = ModelCheckpoint(filepath=modelName+'.h5', monitor='val_loss',verbose=1, save_best_only=True, mode='auto')

print("Start train model")
myModel.fit_generator(
   train_generator,
   steps_per_epoch=qty_train_samples/cfg.model.batchSize,
   epochs=epochs,
   validation_data=valid_generator,
   validation_steps=qty_valid_samples/cfg.model.batchSize,
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