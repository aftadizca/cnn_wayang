from keras.models import model_from_json
import os
from keras import backend as K
import tensorflow.compat.v1 import ConfigProto,InteractiveSession

#PARAMETER
img_width, img_height = 200,200
qty_class = 3
root="traindata"
train_dirs = "traindata/train/"
valid_dirs = "traindata/valid/"
test_dirs = "traindata/test/"
optimizer ='adam'

#label classes
labels = [dirs for path,dirs,filename in os.walk(train_dirs)][0]
#count training sample
def qty_train_samples():
    count = 0
    for path,dirs,filename in os.walk(train_dirs):
        for name in filename:
            count+=1
    print("File in %s = %s" % (train_dirs,count))
    return count
#count validation sample
def qty_valid_samples():
    count = 0
    for path,dirs,filename in os.walk(valid_dirs):
        for name in filename:
            count+=1
    print("File in %s = %s" % (valid_dirs,count))
    return count
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
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
#use CPU, disable GPU
def disableGPU():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
def useGPU(state=True):
    if state:
        GPUConf()
    else:
        disableGPU()
