import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from keras_preprocessing import image
import os
import numpy as np
from param import Config, useGPU
import sys

useGPU(False)

# load json and create model
loaded_model = load_model("model6.h5") or quit()

for i, w in enumerate(loaded_model.get_weights()):
    print(
        "{} -- Total:{}, Zeros: {:.2f}%".format(
            loaded_model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
        )
    )

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_model = converter.convert()

with open("model2.tflite", "wb") as f:
         f.write(tflite_model) 
print("done")