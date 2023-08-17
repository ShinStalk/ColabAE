import os
import sys
from keras.layers import MaxPooling2D, Reshape, Input
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../Utils'))

from Layers import conv2d


class PointNet:
    def __init__(self, input):
        self.input = input

    def build(self):
        num_point = self.input.shape[1]
        point_dim = self.input.shape[2]

        # Encoder
        net = conv2d(self.input, 64, (1, point_dim))
        net = conv2d(net, 64, [1, 1])
        point_feature = conv2d(net, 64, (1, 1))
        net = conv2d(point_feature, 128, (1, 1))
        net = conv2d(net, 1024, (1, 1))

        global_feat = MaxPooling2D(pool_size=(num_point, 1))(net)

        net = Reshape((-1,))(global_feat)

        return Model(inputs=self.input, outputs=net)


if __name__ == "__main__":
    input_tensor = Input(shape=(1024, 3, 1))
    encoder = PointNet(input_tensor)
    encoder.build()