import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.system("cls")

import pandas as pd
import numpy as np

from nltk import MWETokenizer
from config import *

# ? Import ml
import keras
from keras.layers import *

#? Workers
tokenizer = MWETokenizer()

data_frame_train = pd.read_csv(TRAIN_DATA_FILE)
train_data = data_frame_train[["Question", "answer_class"]]

print(train_data)

input_data = []
output_data = []
for string in train_data["Question"]:
    data = tokenizer.tokenize(string)
    input_data.append(data)
for class_str in train_data["answer_class"]:
    output_data.append(class_str)


print(input_data)


# input_data = np.array()
# output_data = np.array()






