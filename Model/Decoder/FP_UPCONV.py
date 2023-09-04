import os
import sys
import tensorflow as tf
from keras import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../../Utils'))

from pointnet_util import PointNetFPModule
from keras.layers import Dense, Dropout, MaxPooling2D


class PointNet2Decoder(Model):
  def __init__(self, encoder, bn_decay=None, **kwargs):
    super(PointNet2Decoder, self).__init__(**kwargs)
    self.encoder = encoder
    self.bn_decay = bn_decay
    self.points1 = [None, None, None, None]
    self.pointnet_fp_layer1 = PointNetFPModule([256,256], scope='fp1', bn_decay=self.bn_decay, bn=True)
    self.pointnet_fp_layer2 = PointNetFPModule([256,256], scope='fp2', bn_decay=self.bn_decay, bn=True)
    self.pointnet_fp_layer3 = PointNetFPModule([256,128], scope='fp3', bn_decay=self.bn_decay, bn=True)
    self.pointnet_fp_layer4 = PointNetFPModule([128,128,128], scope='fp4', bn_decay=self.bn_decay, bn=True)
    self.dense_layer1 = Dense(64, activation='relu')
    self.dense_layer2 = Dense(32, activation='relu')
    self.dense_layer3 = Dense(16, activation='relu')
    self.dense_layer4 = Dense(3, activation='tanh')
    self.dp_layer1 = Dropout(.3)
    self.dp_layer2 = Dropout(.3)
    self.dp_layer3 = Dropout(.2)

    self.max_pooling = MaxPooling2D(pool_size=(self.encoder.num_point, 1))

  def call(self, inputs, **kwargs):
    # Feature Propagation layers
    # self.xyz[1], self.points[1], _ = self.pointnet_sa_layer1(self.xyz[0]) # xyz[1]: (32, 1024, 3), points[1]: (32, 1024, 64)
    # self.xyz[2], self.points[2], _ = self.pointnet_sa_layer2(self.xyz[1]) # xyz[2]: (32, 512, 3), points[2]: (32, 512, 128)
    # self.xyz[3], self.points[3], _ = self.pointnet_sa_layer3(self.xyz[2]) # xyz[3]: (32, 256, 3), points[3]: (32, 256, 256)
    # self.xyz[4], self.points[4], _ = self.pointnet_sa_layer4(self.xyz[3]) # xyz[4]: (32, 128, 3), points[4]: (32, 128, 512)

    self.points1[3] = self.pointnet_fp_layer1(self.encoder.xyz[3], self.encoder.xyz[4], self.encoder.points[3], self.encoder.points[4]) # points1[3]: (32, 256, 256)
    self.points1[2] = self.pointnet_fp_layer2(self.encoder.xyz[2], self.encoder.xyz[3], self.encoder.points[2], self.points1[3]) # points1[2]: (32, 512, 256)
    self.points1[1] = self.pointnet_fp_layer3(self.encoder.xyz[1], self.encoder.xyz[2], self.encoder.points[1], self.points1[2]) # points1[1]: (32, 1024, 128)
    point_features = self.pointnet_fp_layer4(self.encoder.xyz[0], self.encoder.xyz[1], self.encoder.points[0], self.points1[1]) # points1[0]: (32, 1024, 128)


    reshaped_tensor = tf.expand_dims(point_features, axis=2) # (32, 1024, 1, 128)
    
    print(f'global_feat: # {reshaped_tensor.shape}')
    global_feat = self.max_pooling(reshaped_tensor) # (32, 1, 1, 128)
    print(f'global_feat: # {global_feat.shape}')

    net = self.dp_layer1(point_features)
    net = self.dense_layer1(net) # (32, 1024, 64)
    net = self.dp_layer2(net)
    net = self.dense_layer2(net) # (1024, 32)
    net = self.dp_layer3(net)
    net = self.dense_layer3(net) # (32, 1024, 16)

    # Features value normalization
    net = self.normalize(net)

    net = self.dense_layer4(net) # (32, 1024, 3)

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