from keras.models import model_from_json
from keras_preprocessing import image
import os
import numpy as np

 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img_predict = image.load_img(os.getcwd()+"/traindata/test/yudistira2.jpg", target_size=(128,128), color_mode = "grayscale")
img_predict = image.img_to_array(img_predict)
img_predict = np.expand_dims(img_predict/255, axis=0)

result = loaded_model.predict(img_predict)
if result[0][0].round(0) == 1:
    label = "Arjuna"
elif result[0][1].round(0) == 1:
    label = "Bima"
else:
    label = "Yudistira"

print(result.round(4)*100)
print("RESULT : "+label)