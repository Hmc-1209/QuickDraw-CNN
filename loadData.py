"""
    This is a code that been used by training and testing models
    It would automatically download the .npy files needed, if the
    code doesn't work for some reason, please download the files
    from the given link in README.md
"""

import numpy as np

# Datasets
keys = ['cat', 'diamond', 'eye', 'ladder', 'moon']
# keys = ['cat', 'diamond', 'eye', 'ladder', 'moon', 'necklace', 'snowflake', 'sword', 'tornado', 'wine glass']

# Split the list
def split_list(ls, n):
    temp = []
    for index in range(0, len(ls), n):
        temp.append(list(ls[index: index+n]))
    return temp

# Function for loading data
def load(key):
    # If the dataset does not exist, catch the error
    rawDatas = []
    try:
        rawDatas = np.load('./dataset/full_numpy_bitmap_' + key + '.npy')
    except:
        print('Failed to get "full_numpy_bitmap_' + key + '.npy" in dataset folder.')

    data = []
    for rawData in rawDatas[:10000]:
        data.append(split_list(rawData, 28))

    return data

# Functions to be called for getting datas
def loadDatas():
    # Getting datas
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # Variable types for categorical uses
    types = 0
    # Downloading datas from internet

    print('Loading datas ... ')
    for key in keys:
        # Calling load() from loadData.py
        datas = load(key)
        # Split the data into train and test data
        train_data += datas[:8000]
        test_data += datas[8000:]
        # Generate labels for train and test data
        for i in range(8000):
            train_label.append(types)
        for i in range(2000):
            test_label.append(types)
        datas.clear()
        types += 1

    # Checking datas existence
    if len(train_data) != 8000 * len(keys):
        print('Please check out the .npy files properly')
        quit(1)

    print('Loading complete !')
    print('There are ' + str(types) + ' types of images loaded :')
    print(keys)
    return train_data, train_label, test_data, test_label
