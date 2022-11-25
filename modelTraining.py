import loadData as ld
import numpy as np
import matplotlib.pyplot as plt

# Datasets
keys = ['cat', 'diamond', 'eye', 'ladder', 'moon', 'necklace', 'snowflake', 'sword', 'tornado', 'wine glass']

# Getting datas
dataset = []
print('Loading datas ... ')
for key in keys:
    dataset += ld.load(key)
print('Loading complete !')

print(len(dataset))
for i in range(0, 50000, 5100):
    plt.imshow(dataset[i])
    plt.show()
