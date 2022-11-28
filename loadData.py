"""
    This is a code that been used by training and testing models
"""

import os
import urllib.error
import urllib.request

import numpy as np
import random


# Split the list
def split_list(ls, n):
    temp = []
    for index in range(0, len(ls), n):
        temp.append(np.array(ls[index: index + n]))
        # print(ls[index: index+n])
    return temp


# Downloading datas required
def download(keys):
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def download_progress(count, blocksize, totalsize):
        # print('\rDownloading: %5.1f%%' % (count * blocksize * 100.0 / totalsize), end="")
        print(f'\r{icon[count%4]}{count * blocksize * 100.0 / totalsize:5.1f}%', end="")

    for key in keys:
        path = './dataset/full_numpy_bitmap_' + key + '.npy'
        # check whether file exist
        if os.path.exists(path):
            print('%s npy file exists!' % key)
            continue

        print('Downloading %s.npy' % key)
        filename = key.replace('_', '%20') + '.npy'
        key_url = url + filename

        try:
            icon = '⋮⋰⋯⋱'
            urllib.request.urlretrieve(key_url, path, reporthook=download_progress)
            print('')
        except urllib.error.HTTPError:
            print("Failed to get %s npy file." % key)
        except KeyboardInterrupt:
            print("\nDownload suspend.")
            os.remove(path)

    print('Download Finished.')


# Function for loading data
def load(key):
    try:
        raw_datas = np.load('./dataset/full_numpy_bitmap_' + key + '.npy')
    except FileNotFoundError:
        print('Failed to get "full_numpy_bitmap_' + key + '.npy" in dataset folder.')

    data = []
    for rawData in raw_datas[:10000]:
        data.append(split_list(rawData, 28))
    return data


# Function for returning datas
def load_datas(keys):
    # Datas
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # n types of images for one-hot encoding
    types = 0
    print('Loading datas ... ')
    download(keys)
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

    # Shuffle datas and labels
    pack_train = list(zip(train_data, train_label))
    random.shuffle(pack_train)
    pack_test = list(zip(test_data, test_label))
    random.shuffle(pack_test)
    train_data, train_label = zip(*pack_train)
    test_data, test_label = zip(*pack_test)

    # Convert lists into ndarray
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_data, train_label, test_data, test_label
