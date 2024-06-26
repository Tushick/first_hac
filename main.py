import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.system("cls")

import pandas as pd
import numpy as np


from functions.worker_data import *
from config import *

# ? Import ml
import keras
from keras.layers import *

data_frame_train = pd.read_csv(TRAIN_DATA_FILE)
train_data = data_frame_train[["Question", "answer_class"]]

input_data = []
output_data = []
max_len_data = 0
for string in train_data["Question"]:
    data = tokenize_text(string)

    if len(data) > max_len_data:
        max_len_data = len(data)

    input_data.append(data)
for class_str in train_data["answer_class"]:
    output_data.append(class_str)

for index, arr in enumerate(input_data):
    while len(arr) < max_len_data:
        arr.append(-1)
    input_data[index] = arr
    


input_data = np.array(input_data)
output_data = np.array(output_data)

model = keras.Sequential()
# model.add([
#     Dense(units=1),
# ])
model.add(Dense(units=1, activation='relu'))


model.compile(loss='mse', optimizer='sgd')

fit_result = model.fit(input_data, 
                       output_data, 
                       epochs=200
                        )


if not os.path.isdir(MODELS_SAVE_DIR): os.mkdir(MODELS_SAVE_DIR)

model.save(MODEL_PATH)
