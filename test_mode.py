import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.system("cls")

import numpy as np
from keras.models import load_model
from config import *
from functions.worker_data import *

model = load_model(MODEL_PATH)

text = 'Hello'

data_text = np.array(tokenize_text(text))

predict = model.predict(data_text)
