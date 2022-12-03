"""
    This is a code that been used by training and testing models
"""

import os
import urllib.error
import urllib.request
import ssl
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

# Downloading datas required
def download(keys):
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def download_progress(count, blocksize, totalsize):
        now = count * blocksize / totalsize
        total_blocks = 50

        if now >= 1.0:
            print(
                f'\r[{"█" * total_blocks}] {100}%', end='')  # 輸出不換行的內容
        else:
            plus_block = int(now * total_blocks)
            minus_block = int(total_blocks - plus_block)
            percentage = round(now * 100, 2)
            print(f'\r[{plus_block * "█"}{minus_block * " "}] {percentage}%', end='')  # 輸出不換行的內容

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
def load(key, amount):
    try:
        raw_datas = np.load('./dataset/full_numpy_bitmap_' + key + '.npy')
    except FileNotFoundError:
        print('Failed to get "full_numpy_bitmap_' + key + '.npy" in dataset folder.')
        return

    return raw_datas[:amount].reshape(amount, 28, 28)


# Function for returning datas
def load_datas(keys, amount, test_split=0.2):
    # n types of images for one-hot encoding
    types = 0

    test_num = int(amount * test_split)
    train_num = int(amount - test_num)

    # Datas
    train_data = np.empty((train_num * len(keys), 28, 28))
    train_label = np.empty(train_num * len(keys))
    test_data = np.empty((test_num * len(keys), 28, 28))
    test_label = np.empty(test_num * len(keys))

    print('Loading datas ... ')
    download(keys)

    for key in keys:
        datas = load(key, amount)

        start_train = types * train_num
        start_test = types * test_num

        # Checking datas existence
        if datas is None:
            np.delete(train_data, np.s_[start_train:start_train + train_num], axis=0)
            np.delete(test_data, np.s_[start_test:start_test + test_num], axis=0)
            del keys[types]
            continue

        # Split the data into train and test data
        train_data[start_train:start_train + train_num] = datas[:train_num]
        test_data[start_test:start_test + test_num] = datas[train_num:]

        train_label[start_train:start_train + train_num] = types
        test_label[start_test:start_test + test_num] = types

        types += 1

    print('Loading complete !')
    print('There are ' + str(types) + ' types of images loaded :' + '\n' + str(keys))
    print(f'Train datas : {str(len(train_data))} | Test datas : {str(len(test_data))}', end='\n\n')

    # Shuffle datas and labels
    shuffle_ix = np.random.permutation(np.arange(len(train_data)))
    train_data = np.array(train_data)[shuffle_ix]
    train_label = np.array(train_label)[shuffle_ix]

    return train_data, train_label, test_data, test_label
