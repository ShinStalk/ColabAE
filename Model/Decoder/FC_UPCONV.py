import os
import sys
import tensorflow as tf
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../Utils'))

from tf_util import fully_connected_v2, conv2d_transpose_v2
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

        self.upconv1 = conv2d_transpose_v2(512, kernel_size=(2, 2), stride=(1, 1), padding='VALID', scope='upconv1', bn=True, bn_decay=bn_decay)
        self.upconv2 = conv2d_transpose_v2(256, kernel_size=(3, 3), stride=(1, 1), padding='VALID', scope='upconv2', bn=True, bn_decay=bn_decay)
        self.upconv3 = conv2d_transpose_v2(128, kernel_size=(3, 3), stride=(1, 1), padding='VALID', scope='upconv3', bn=True, bn_decay=bn_decay)
        self.upconv4 = conv2d_transpose_v2(64, kernel_size=(6, 6), stride=(2, 2), padding='VALID', scope='upconv4', bn=True, bn_decay=bn_decay)
        self.upconv5 = conv2d_transpose_v2(6, kernel_size=(1, 1), stride=(1, 1), padding='VALID', scope='upconv5', activation_fn=None)


    def call(self, xyz_input, points_input, **kwargs):

        embedding = self.fc1(points_input) # (None, 512)

        # FC Decoder
        net = self.fc2(embedding)  # (None, 512)
        net = self.fc3(net)  # (None, 512)
        net = self.fc4(net)  # (None, 1536)
        pc_fc = tf.reshape(net, [-1, 512, 3])  # (None, 512, 3)

        # UPCONV Decoder
        net = tf.reshape(embedding, [-1, 1, 1, embedding.shape[1]])  # (None, 1, 1, 512)
        net = self.upconv1(net)  # (32, 2, 2, 512)
        net = self.upconv2(net)  # (32, 4, 4, 256)
        net = self.upconv3(net)  # (32, 6, 6, 128)
        net = self.upconv4(net)  # (32, 16, 16, 64)
        net = self.upconv5(net)  # (32, 16, 16, 6)
        pc_upconv = tf.reshape(net, [-1, 512, 3])  # (32, 512, 3)

        # Set union
        net = tf.concat(values=[pc_fc, pc_upconv], axis=1)
        print(f'concat: {net.shape}')  #  (32, 1024, 3)

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