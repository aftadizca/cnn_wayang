from keras.models import model_from_json, load_model
from keras_preprocessing import image
import os
import numpy as np
from param import Config, useGPU
import sys

cfg = Config()

useGPU(False)
# load json and create model
loaded_model = load_model("model3-all.h5") or quit()

#loaded_model.save('model3-all')

for p,d,f in os.walk(cfg.dirs.test_path):
    for name in f:
        print("")
        img_path=os.path.join(p,name)
        print(img_path.upper())
        img_predict = image.load_img(img_path, target_size=(cfg.img.width, cfg.img.width), color_mode = cfg.img.color_mode)
        img_predict = image.img_to_array(img_predict)
        img_predict = np.expand_dims(img_predict/255, axis=0)

        result = loaded_model.predict(img_predict)
        print(result)
        for x in range(0,3):
            print("%s : %.3f%%" % (cfg.dirs.labels[x].upper(),result[0][x]*100))
