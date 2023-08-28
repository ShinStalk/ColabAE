import os
import sys
from keras.layers import Reshape, Dense, Input
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../Utils'))

from tf_util import tf2_fully_connected


class FC:
    def __init__(self):
        self.num_point = None

    def build(self, input_shape):
        print(f'[build] FC input_shape: {input_shape}')
        self.num_point = input_shape[1]

    def call(self, inputs, **kwargs):
        # FC Decoder
        net = tf2_fully_connected(inputs, 1024)
        net = tf2_fully_connected(net, 1024)
        net = Dense(self.num_point * 3)(net)

        net = Reshape((self.num_point, 3))(net)

        return Model(inputs=inputs, outputs=net)