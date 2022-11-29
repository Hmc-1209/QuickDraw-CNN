import loadData
import numpy as np

keys = ['angel', 'ant', 'apple']
loadData.download(keys)

arr1 = loadData.load('angle')
print(arr1.shape)