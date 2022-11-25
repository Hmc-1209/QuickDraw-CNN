import loadData as ld
import numpy as np
import matplotlib.pyplot as plt

# Datasets
keys = ['cat', 'diamond', 'eye', 'ladder', 'moon']
# keys = ['cat', 'diamond', 'eye', 'ladder', 'moon', 'necklace', 'snowflake', 'sword', 'tornado', 'wine glass']

# Getting datas
train_data = []
train_label = []
test_data = []
test_label = []
print('Loading datas ... ')
for key in keys:
    # Calling load() from loadData.py
    datas = ld.load(key)
    # Split the data into train and test data
    train_data += datas[:8000]
    test_data += datas[8000:]
    # Generate labels for train and test data
    for i in range(8000):
        train_label.append(key)
    for i in range(2000):
        test_label.append(key)
    datas.clear()
print('Loading complete !')


