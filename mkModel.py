import numpy as np
import tensorflow as tf
from math import ceil


def activate(x, activation=None):
    if activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(x))
    elif activation == 'softmax':
        return np.exp(x) / sum(np.exp(x))
    else:
        if activation is not None:
            print('No such activation function!')
        return x


def get_output_shape(self, in_height, in_width):
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
    return out_height, out_width


def expand_x(self, x, shape):
    pad_h = (shape[0] - self.filter_h[0]) / 2
    pad_w = (shape[1] - self.filter_w[1]) / 2
    return np.pad(x, (pad_h, pad_w), mode='constant')


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Model:
    def __init__(self, layers=None):
        self.layers = layers if layers else []
        self.weight = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, x, y, epochs, batch_size):
        for layer in self.layers:
            y_ = layer.get_output(x)
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
        self.activation = activation
        self.weights = kwargs['weight']
        self.bias = kwargs['bias']

    def check_data(self):
        return

    def get_output(self, x: np.ndarray):
        batch_size, in_height, in_width, in_depth = x.shape

        self.weights = self.weights if self.weights else \
            np.random.randn(self.filter_h, self.filter_w, in_depth, self.filters) * np.sqrt(2 / self.filter_h)

        # 維持原輸入尺寸
        if self.padding == 'same':
            expand_x(self, x, (in_height, in_width))

        out_height, out_width = get_output_shape(self, in_height, in_width)
        f_map = np.empty((batch_size, out_height, out_width, self.filters))

        for batch in range(batch_size):
            for f in range(self.filters):
                mask = self.weights[:, :, :, f].reshape(self.filter_h, self.filter_w)
                f_map_h = 0
                f_map_w = 0
                for h in range(0, out_height, self.strides[0]):
                    for w in range(0, out_width, self.strides[1]):
                        x_v = x[batch, h:h + self.filter_h, w: w + self.filter_w]
                        f_map[batch, f_map_h, f_map_w, f] = np.matmul(x_v, mask)
                        f_map_w += 1
                    f_map_h += 1

        if self.bias is None:
            f_map += f_map + self.bias
        f_map = activate(f_map, self.activation)

        return f_map


class MaxPooling2D:
    def __init__(self, pool_size, strides=None, padding='valid'):
        self.filter_h = pool_size[0]
        self.filter_w = pool_size[1]
        self.strides = strides
        self.padding = padding

    def get_output(self, x):
        batch_size, in_height, in_width, in_depth = x.shape

        if self.padding == 'same':
            expand_x(self, x, (in_height, in_width))

        out_height, out_width = get_output_shape(self, in_height, in_width)
        f_map = np.empty((batch_size, out_height, out_width, in_depth))

        for batch in range(batch_size):
            for d in range(in_depth):
                f_map_h = 0
                f_map_w = 0
                for h in range(0, out_height, self.strides[0]):
                    for w in range(0, out_width, self.strides[1]):
                        x_v = x[batch, h:h + self.filter_h, w: w + self.filter_w]
                        f_map[batch, f_map_h, f_map_w, d] = np.max(x_v)
                        f_map_w += 1
                    f_map_h += 1

        return


class Fatten:
    def get_output(self, x):
        batch_size, in_height, in_width, in_depth = x.shape

        f_map = None
        for batch in range(batch_size):
            for h in range(in_height):
                for w in range(in_width):
                    if f_map is None:
                        f_map = x[batch, h, w]
                        continue
                    f_map = np.concatenate((f_map, x[batch, h, w]))

        return f_map.reshape(batch_size, in_height*in_width*in_depth)
