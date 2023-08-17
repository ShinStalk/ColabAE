import os
import sys
from keras.layers import Reshape, Dense, Input
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../Utils'))

from Layers import fully_connected


class FC:
    def __init__(self, input):
        self.input = input

    def build(self):
        num_point = self.input.shape[1]

        # FC Decoder
        net = fully_connected(self.input, 1024)
        net = fully_connected(net, 1024)
        net = Dense(num_point * 3)(net)

        net = Reshape((num_point, 3))(net)

        return Model(inputs=self.input, outputs=net)

if __name__ == "__main__":
    input_tensor = Input(shape=(1024))
    encoder = FC(input_tensor)
    encoder.build()