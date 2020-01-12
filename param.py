from keras.models import model_from_json
import os
from keras import backend as K
from tensorflow.compat.v1 import ConfigProto,InteractiveSession
import tensorflow as tf

#PARAMETER
img_width, img_height = 200,200
qty_class = 3
root="traindata"
train_dirs = "traindata/train/"
valid_dirs = "traindata/valid/"
test_dirs = "traindata/test/"
optimizer ='adam'

class Config:
    def __init__(self):
        self.model = self.Models()
        self.img = self.Images()
        self.dirs = self.Dirs()

    class Models:
        def __init__(self):
            self.qty_class = 3
            self.optimizer = 'adam'
            self.batchSize = 10

    class Images:
        def __init__(self):
            self.width = 200
            self.height = 200
            self.color_mode = 'grayscale'

    class Dirs:
        def __init__(self):
            self.train_path = 'traindata/train'
            self.validation_path = 'traindata/valid'
            self.test_path = 'traindata/test'
            self.log_path = 'logs/scalars/'
            self.labels = [dirs for path,dirs,filename in os.walk(self.train_path)][0]
            self.class_count = len(self.labels)

#count validation sample
def count_files(path):
    try:
        count = 0
        for _,_,filename in os.walk(path):
            for _ in filename:
                count+=1
        print("File in %s = %s" % (path,count))
        return count
    except:
        return 0
#loading model, return False when no model file found
def load_model(filename):
    try: 
        # load json and create model
        with open(filename+'.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(filename+".h5")
            print("Loaded "+filename+" from disk")
            
            print("Compile Model")
            loaded_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
            )
            return loaded_model
    except FileNotFoundError:
        print(filename+" not found!")
        return False
#GPU config
def GPUConf():
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    #K.set_session(session)
#use CPU, disable GPU
def disableGPU():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#disable or enable GPU. set False to disable GPU
def useGPU(state=True):
    if state:
        GPUConf()
    else:
        disableGPU()
