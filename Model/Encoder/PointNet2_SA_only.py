import os
import sys
import tensorflow as tf
from keras.layers import MaxPooling2D, Reshape, Input
from keras import Model

BASE_DIR = os.path.abspath('')
sys.path.append(os.path.join(BASE_DIR, 'ColabAE/Utils'))

from pointnet_util import pointnet_sa_module


class PointNet2Encoder:
    def __init__(self, input):
        self.input = input

    def build(self):
        num_point = self.input.shape[1]
        point_dim = self.input.shape[2]

        print(f'input.shape: {self.input.shape}')  # logs: (None, 1024, 3)
        l0_xyz = tf.reshape(self.input, [BATCH_SIZE, num_point, point_dim])
        print(f'l0_xyz.shape: {l0_xyz.shape}')  # logs: (32, 1024, 3)
        l0_points = None

        ##########
        # Encoder
        ##########
        # # Set abstraction layers
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64, 64, 128], mlp2=None, group_all=False,
                                                           is_training=True, bn_decay=0.9, scope='layer1', use_nchw=False)
        print(f'l1_xyz.shape: {l1_xyz.shape}')  # logs: (32, 512, 3)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=256, radius=0.4, nsample=64, mlp=[128, 128, 256], mlp2=None, group_all=False,
                                                           is_training=True, bn_decay=0.9, scope='layer2')
        print(f'l2_xyz.shape: {l2_xyz.shape}')  # logs: (32, 256, 3)
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=0.8, nsample=128, mlp=[256, 512, 1024], mlp2=None, group_all=False,
                                                           is_training=True, bn_decay=0.9, scope='layer3')
        print(f'l3_xyz.shape: {l3_xyz.shape}')  # logs: (32, 128, 3)

        # return Model(inputs=self.input, outputs=net)


# use example: PointNet2Encoder(Input(shape=(1024, 3))).build()