"""
    This is a code that been used by training and testing models
"""

import os
import urllib.error
import urllib.request

import numpy as np

# Datasets
keys = ['cat', 'diamond', 'eye', 'ladder', 'moon']


# keys = ['cat', 'diamond', 'eye', 'ladder', 'moon', 'necklace', 'snowflake', 'sword', 'tornado', 'wine glass']

# Split the list
def split_list(ls, n):
    temp = []
    for index in range(0, len(ls), n):
        temp.append(list(ls[index: index + n]))
        # print(ls[index: index+n])
    return temp


# Downloading datas required
def download():
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def download_progress(a, b, c):
        print('\rDownloading: %5.1f%%' % (a * b * 100.0 / c), end="")

    for key in keys:
        path = './dataset/full_numpy_bitmap_' + key + '.npy'
        # check whether file exist
        if os.path.exists(path):
            # print('%s file exists!' % key)
            continue

        print('Downloading %s npy file' % key)
        k_url = key.replace('_', '%20')
        key_url = url + k_url + '.npy'

        try:
            urllib.request.urlretrieve(key_url, path, reporthook=download_progress)
            print('')
        except urllib.error.HTTPError:
            print("No such file")


# Function for loading data
def load(key):
    raw_datas = []
    try:
        raw_datas = np.load('./dataset/full_numpy_bitmap_' + key + '.npy')
    except FileNotFoundError:
        print('Failed to get "full_numpy_bitmap_' + key + '.npy" in dataset folder.')

    data = []
    for rawData in raw_datas[:10000]:
        data.append(split_list(rawData, 28))
    return data


# Function for returning datas
def load_datas():
    # Datas
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # n types of images for one-hot encoding
    types = 0
    print('Loading datas ... ')
    download()
    for key in keys:
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
    print('There are ' + str(types) + ' types of images loaded :' + '\n' + str(keys))
    print('Train datas : ' + str(len(train_data)), 'Test datas : ' + str(len(test_data)) + '\n')
    return train_data, train_label, test_data, test_label
