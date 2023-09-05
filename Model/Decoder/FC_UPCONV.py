import os
import sys
import tensorflow as tf
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../Utils'))

from pointnet_util import PointNetFPModule
from tf_util import fully_connected_v2
from keras.layers import Dense, Dropout, MaxPooling2D


class PointNet2Decoder(Model):
    def __init__(self, encoder, bn_decay=None, **kwargs):
        super(PointNet2Decoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.bn_decay = bn_decay
        self.points1 = [None, None, None, None]

        self.fc1 = fully_connected_v2(512)
        self.fc2 = fully_connected_v2(512)
        self.fc3 = fully_connected_v2(512)
        self.fc4 = fully_connected_v2(512 * 3, activation_fn=None)


    def call(self, xyz_input, points_input, **kwargs):

        embedding = self.fc1(512) # (None, 512)

        # FC Decoder
        net = self.fc2(embedding)  # (None, 512)
        net = self.fc3(net)  # (None, 512)
        net = self.fc4(net)  # (None, 1536)
        pc_fc = tf.reshape(net, [-1, -1, 3])  # (None, 512, 3)

        # UPCONV Decoder
        net = tf.reshape(embedding, [1, 1, -1])  # (None, 1, 1, 512)
        net = conv2d_transpose(net, 512, kernel_size=(2, 2), stride=(1, 1))  # Output shape: (None, 2, 2, 512)
        net = conv2d_transpose(net, 256, kernel_size=(3, 3), stride=(1, 1))  # Output shape: (None, 4, 4, 256)
        net = conv2d_transpose(net, 256, kernel_size=(4, 4), stride=(2, 2))  # Output shape: (None, 10, 10, 256)
        net = conv2d_transpose(net, 128, kernel_size=(5, 5), stride=(3, 3))  # Output shape: (None, 32, 32, 128)
        net = Conv2DTranspose(3, (1, 1), strides=(1, 1))(net) # Output shape: (None, 32, 32, 3)

        # TODO : Remove these two lines after getting the last Conv2DTranspose layer compatible with Reshape((half_num_point, 3))(net)
        # The following two layers are added
        net = Flatten()(net) # (None, 3072)
        net = Dense(half_num_point * 3)(net) # (None, 1536)
        pc_upconv = Reshape((half_num_point, 3))(net) # (None, 512, 3)


        return net

    def print_tensor(self, tensor, name):
        # Usage: Lambda(lambda x: self.print_tensor(x, 'TensorName'))(tensor)
        tf.print(f'\n{name} Value:', tensor)
        return tensor

    def normalize(self, tensor):
        min_vals = tf.reduce_min(tensor, axis=-1, keepdims=True)
        max_vals = tf.reduce_max(tensor, axis=-1, keepdims=True)
        scaled_tensor = (tensor - min_vals) / (max_vals - min_vals)
        # scaled_tensor = (scaled_tensor * 2) - 1 # Rescale to [-1, 1] (?)
        return scaled_tensor