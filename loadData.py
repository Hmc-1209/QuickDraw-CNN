"""
    This is a code that been used by training and testing models
"""

import os
import urllib.error
import urllib.request

import numpy as np


# Downloading datas required
def download(keys):
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def download_progress(count, blocksize, totalsize):
        print(f'\r{icon[count % 4]}{count * blocksize * 100.0 / totalsize:5.1f}%', end="")

    for key in keys:
        path = './dataset/full_numpy_bitmap_' + key + '.npy'
        # check whether file exist
        if os.path.exists(path):
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

    return raw_datas[:10000].reshape(10000, 28, 28)


# Function for returning datas
def load_datas(keys):
    # n types of images for one-hot encoding
    types = 0

    # Datas
    train_data = np.empty((8000 * len(keys), 28, 28))
    train_label = np.empty(8000 * len(keys))
    test_data = np.empty((2000 * len(keys), 28, 28))
    test_label = np.empty(2000 * len(keys))

    print('Loading datas ... ')
    download(keys)

    for key in keys:
        datas = load(key)

        # Split the data into train and test data
        start_train = types * 8000
        start_test = types * 2000

        train_data[start_train:start_train + 8000] = datas[:8000]
        test_data[start_test:start_test + 2000] = datas[8000:]

        train_label[start_train:start_train + 8000] = types
        test_label[start_test:start_test + 2000] = types

        types += 1

    # Checking datas existence
    if len(train_data) != 8000 * len(keys):
        print('Please check out the .npy files properly')
        raise SystemExit(1)

    print('Loading complete !')
    print('There are ' + str(types) + ' types of images loaded :' + '\n' + str(keys))
    print('Train datas : ' + str(len(train_data)), 'Test datas : ' + str(len(test_data)) + '\n')

    # Shuffle datas and labels
    shuffle_ix = np.random.permutation(np.arange(len(train_data)))
    train_data = np.array(train_data)[shuffle_ix]
    train_label = np.array(train_label)[shuffle_ix]

    shuffle_iy = np.random.permutation(np.arange(len(test_data)))
    test_data = np.array(test_data)[shuffle_iy]
    test_label = np.array(test_label)[shuffle_iy]

    return train_data, train_label, test_data, test_label
