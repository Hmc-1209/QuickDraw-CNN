"""
    This is the code for training model

"""

from keras import Sequential
from keras.utils import to_categorical
import loadData as lD
import numpy as np
import matplotlib.pyplot as plt

# Getting datas
train_data, train_label, test_data, test_label = lD.load_datas()


# Data Preprocessing
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)

# Creating models
model = Sequential()
