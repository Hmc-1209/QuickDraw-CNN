"""
    This is a code that been used by training and testing models
"""

import numpy as np


# Split the list
def split_list(ls, n):
    temp = []
    for index in range(0, len(ls), n):
        temp.append(list(ls[index: index+n]))
        # print(ls[index: index+n])
    return temp

# Function for loading data
def load(key):
    rawDatas = np.load('./dataset/full_numpy_bitmap_'+key+'.npy')
    data = []
    for rawData in rawDatas[:10000]:
        data.append(split_list(rawData, 28))

    return data
