import loadData as ld
import numpy as np
import matplotlib.pyplot as plt

# Datasets
keys = ['cat']

# Getting datas
dataset = []
for key in keys:
    dataset += ld.load(key)[:1000]

print(len(dataset))
# plt.imshow()
# plt.show()
