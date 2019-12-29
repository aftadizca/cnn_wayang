from keras.models import model_from_json
from keras_preprocessing import image
import os
import numpy as np
from param import img_height,img_width,load_model,GPUConf,test_dirs,labels,\
    useGPU
import sys


useGPU(False)
# load json and create model
loaded_model = load_model("model3") or quit()

for p,d,f in os.walk(test_dirs):
    for name in f:
        print("")
        img_path=os.path.join(p,name)
        print(img_path.upper())
        img_predict = image.load_img(img_path, target_size=(img_width,img_height), color_mode = "grayscale")
        img_predict = image.img_to_array(img_predict)
        img_predict = np.expand_dims(img_predict/255, axis=0)

        result = loaded_model.predict(img_predict)
        print(result)
        for x in range(0,3):
            print("%s : %.3f%%" % (labels[x].upper(),result[0][x]*100))
