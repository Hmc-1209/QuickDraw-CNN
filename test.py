import loadData
import numpy as np

keys = ['angel', 'ant']
loadData.download(keys)

loadData.load_datas(keys)

# arr1 = [1, 2, 3, 4, 5]
# arr2 = [6, 7, 8, 9, 10]
#
# shuffle_ix = np.random.permutation(np.arange(len(arr1)))
# print(shuffle_ix)
# arr1 = np.array(arr1)[shuffle_ix]
# arr2 = np.array(arr2)[shuffle_ix]
#
# print(arr1)
# print(arr2)