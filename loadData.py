"""
    This is a code that been used by training and testing models
"""

import numpy as np
from urllib.request import urlretrieve
import os


# Split the list
def split_list(ls, n):
    temp = []
    for index in range(0, len(ls), n):
        temp.append(list(ls[index: index + n]))
        # print(ls[index: index+n])
    return temp


def download(keys):
    url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

    def download_progress(a, b, c):
        print('\rDownloading: %5.1f%%' % (a * b * 100.0 / c), end="")

    for key in keys:
        path = './dataset/full_numpy_bitmap_' + key + '.npy'
        # check whether file exist
        if os.path.exists(path):
            print('%s file exists!' % key)
            return

        print('Downloading %s npy file' % key)
        k_url = key.replace('_', '%20')
        key_url = url + k_url + '.npy'

        try:
            urlretrieve(key_url, path, reporthook=download_progress)
        except:
            print("No such file")


# Function for loading data
def load(key):
    rawDatas = np.load('./dataset/full_numpy_bitmap_' + key + '.npy')
    data = []
    for rawData in rawDatas[:10000]:
        data.append(split_list(rawData, 28))

    return data
