import pandas as pd
import numpy as np
from config import *

data_frame = pd.read_csv(TRAIN_DATA_FILE)
train_data = data_frame[["Question", "answer_class"]]

print(train_data)
