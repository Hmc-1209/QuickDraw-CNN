import numpy as np
from math import ceil


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def activate(x, activation=None):
    if activation == 'relu':
        return relu(x)
    elif activation == 'sigmoid':
        return sigmoid(x)
    elif activation == 'softmax':
        return softmax(x)
    else:
        if activation is not None:
            print('No such activation function!')
        return x


class Model:
    def __init__(self, layers=None):
        self.layers = layers if layers else []
        self.weight = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, epochs, batch_size):
        return


# depth 3為RGB 1為灰階
# depth RGB 未實現
class Conv2D:
    def __init__(self, filters: int, kernel_size, strides=(1, 1), padding='valid', activation=None, **kwargs):
        self.filters = filters
        self.filter_h = kernel_size[0]
        self.filter_w = kernel_size[1]
        self.strides = strides
        self.padding = padding
        # self.input_shape = kwargs['input_shape']
        self.activation = activation
        self.weights = kwargs['weight']
        self.masks = self.weights if self.weights \
            else np.random.randn(filters, kernel_size[0], kernel_size[1]) * np.sqrt(2 / kernel_size[0])

        # if self.input_shape:
        #     self.in_height, self.in_width, self.in_depth = self.input_shape

    def check_data(self):
        return

    def get_output_shape(self, in_height, in_width, in_depth):
        if in_depth == 3:
            return

        out_depth = self.filters
        if self.padding == 'same':
            out_height = ceil(float(in_height) / float(self.strides[0]))
            out_width = ceil(float(in_width) / float(self.strides[1]))
        elif self.padding == 'valid':
            out_height = ceil(float(in_height - self.filter_h + 1) / float(self.strides[0]))
            out_width = ceil(float(in_width - self.filter_w + 1) / float(self.strides[1]))
        else:
            print("padding should be valid or same.")
            raise ValueError(
                "Invalid value for argument `self.paddings`. "
            )
        return out_height, out_width, out_depth

    def get_output(self, x: np.ndarray):
        batch_size, in_height, in_width, in_depth = x.shape

        # 維持原輸入尺寸
        if self.padding == 'same':
            pad_h = (in_height - self.filter_h) / 2
            pad_w = (in_width - self.filter_w) / 2
            x = np.pad(x, (pad_h, pad_w), mode='constant')

        out_height, out_width, _ = self.get_output_shape(in_height, in_width, in_depth)
        f_map = np.empty((batch_size, out_height, out_width, self.filters))

        for batch in range(batch_size):
            for f in range(self.filters):
                mask = self.masks[f]
                f_map_h = 0
                f_map_w = 0
                for h in range(0, out_height, self.strides[0]):
                    for w in range(0, out_width, self.strides[1]):
                        x_v = x[batch, h:h + self.filter_h, w: w + self.filter_w]
                        f_map[batch, f_map_h, f_map_w, f] = np.matmul(x_v, mask)
                        f_map_w += 1
                    f_map_h += 1

        f_map = activate(f_map, self.activation)

        return f_map


class MaxPooling2D:
    def __init__(self, pool_size, strides=None, padding='valid'):
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def get_output(self, x):
        return
