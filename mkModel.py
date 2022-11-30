import numpy as np
from math import ceil


def relu(inputs):
    return np.maximum(0, inputs)

def sigmoid(inputs):
    return 1 / (1 + np.exp(inputs))

def softmax(inputs):
    return np.exp(inputs)/sum(np.exp(inputs))


class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, epochs, batch_size):
        return


# depth 3為RGB 1為灰階
# depth RGB 未實現
class Conv2D:
    def __init__(self, filters: int, kernel_size, input_shape, strides: int = (1, 1), padding='valid',
                 activation=None):
        self.filters = filters
        self.filter_h = kernel_size[0]
        self.filter_w = kernel_size[1]
        self.strides = strides
        self.padding = padding
        self.input_shape = input_shape
        self.activation = activation
        self.masks = np.random.randn(filters, kernel_size[0], kernel_size[1])

    def get_output_shape(self):
        in_height, in_width, _ = self.input_shape
        out_depth = self.filters
        if self.padding == 'same':
            out_height = ceil(float(in_height) / float(self.strides[0]))
            out_width = ceil(float(in_width) / float(self.strides[1]))
        elif self.padding == 'valid':
            out_height = ceil(float(in_height - self.filter_h + 1) / float(self.strides[0]))
            out_width = ceil(float(in_width - self.filter_w + 1) / float(self.strides[0]))
        else:
            print("padding should be valid or same.")
            raise SystemExit(1)
        return out_height, out_width, out_depth

    def get_feature_map(self, x):
        # 維持原輸入尺寸
        if self.padding == 'same':
            return

        out_height, out_width, _ = self.get_output_shape()
        f_map = np.empty((out_height, out_width, self.filters))

        for f in range(self.filters):
            mask = self.masks[f]
            f_map_h = 0
            f_map_w = 0
            for h in range(0, out_height, self.strides):
                f_map_h += 1
                for w in range(0, out_width, self.strides):
                    f_map_w += 1
                    x_v = x[h:h + self.filter_h, w: w + self.filter_w]
                    f_map[f_map_h, f_map_w, f] = np.matmul(x_v, mask)

        if self.activation == 'relu':
            f_map = relu(f_map)
        elif self.activation == 'sigmoid':
            f_map = sigmoid(f_map)
        elif self.activation == 'softmax':
            f_map = softmax(f_map)
        else:
            print('No such activation function!')
            raise SystemExit(1)

        return f_map


class MaxPooling2D:
    def __init__(self, pool_size, strides=None, padding='valid'):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
