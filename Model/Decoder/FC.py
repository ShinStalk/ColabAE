import os
import sys
from keras.layers import Reshape, Dense, Input, Flatten
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../Utils'))

from tf_util import fully_connected_v2


class FC(Model):
    def __init__(self, point_dim, **kwargs):
        super(FC, self).__init__(**kwargs)
        self.flatten_layer = Flatten()
        self.fc_layer1 = fully_connected_v2(256*point_dim, bn=True)
        self.fc_layer2 = fully_connected_v2(512*point_dim, bn=True)
        self.fc_layer3 = fully_connected_v2(1024*point_dim, activation_fn=None)


    def call(self, inputs, **kwargs):
        # inputs: (32, 128, 3)
        # FC Decoder
        net = self.flatten_layer(inputs) # (32, 384)
        net = self.fc_layer1(net) # (32, 768)
        net = self.fc_layer2(net) # (32, 1536)
        net = self.fc_layer3(net) # (32, 3072)

        net = Reshape((1024, 3))(net) # (32, 1024, 3)

        return net