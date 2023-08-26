import glob
import os
import sys
import numpy as np

NB_OF_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001

# Loading the point clouds data
point_cloud_files = glob.glob(os.path.join("Data/PointClouds"+str(NB_OF_POINTS), "*.npy"))
point_clouds = [np.load(pc_file) for pc_file in point_cloud_files]
point_clouds = np.stack(point_clouds)

import tensorflow as tf
from keras import Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BASE_DIR = os.path.abspath('')
sys.path.append(os.path.join(BASE_DIR, 'Utils'))
sys.path.append(os.path.join(BASE_DIR, 'Loss'))

from pointnet_util import pointnet_sa_module, pointnet_fp_module, PointNetSAModule
from tf_util import fully_connected_v2

class PointNet2Encoder(Model):
  def __init__(self, input_shape, latent_dim, bn_decay=None, **kwargs):
    super(PointNet2Encoder, self).__init__(**kwargs)
    self.num_point = input_shape[0]
    self.point_dim = input_shape[1]
    self.latent_dim = latent_dim
    self.bn_decay = bn_decay
    self.xyz = [None, None, None, None, None]
    self.points = [None, None, None, None, None]
    self.pointnet_sa_layer1 = PointNetSAModule(self.points[0], npoint=1024, radius=0.1, nsample=32, mlp=[32,32,64], scope='sa1', bn_decay=self.bn_decay)
    self.pointnet_sa_layer2 = PointNetSAModule(self.points[1], npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], scope='sa2', bn_decay=self.bn_decay)
    self.pointnet_sa_layer3 = PointNetSAModule(self.points[2], npoint=256, radius=0.4, nsample=32, mlp=[128,128,256], scope='sa3', bn_decay=self.bn_decay)
    self.pointnet_sa_layer4 = PointNetSAModule(self.points[3], npoint=128, radius=0.8, nsample=32, mlp=[256,256,512], scope='sa4', bn_decay=self.bn_decay)


  def build(self, input_shape):
    self.pointnet_sa_layer1.build(input_shape)
    self.pointnet_sa_layer2.build(input_shape)
    self.pointnet_sa_layer3.build(input_shape)
    self.pointnet_sa_layer4.build(input_shape)

  def call(self, inputs, **kwargs):
    self.xyz[0] = tf.reshape(inputs, [-1, self.num_point, self.point_dim])


    self.xyz[1], self.points[1], _ = self.pointnet_sa_layer1(self.xyz[0])
    # print(f'l1_xyz: {self.xyz[1].shape}, l1_points: {self.points[1].shape}') # l1_xyz: (32, 1024, 3), l1_points: (32, 1024, 64)
    self.xyz[2], self.points[2], _ = self.pointnet_sa_layer2(self.xyz[1])
    # print(f'l2_xyz: {self.xyz[2].shape}, l2_points: {self.points[2].shape}') # l2_xyz: (32, 512, 3), l2_points: (32, 512, 128)
    self.xyz[3], self.points[3], _ = self.pointnet_sa_layer3(self.xyz[2])
    # print(f'l3_xyz: {self.xyz[3].shape}, l3_points: {self.points[3].shape}') # l3_xyz: (32, 256, 3), l3_points: (32, 256, 256)
    self.xyz[4], self.points[4], _ = self.pointnet_sa_layer4(self.xyz[3])
    # print(f'l4_xyz: {self.xyz[4].shape}, l4_points: {self.points[4].shape}') # l4_xyz: (32, 128, 3), l4_points: (32, 128, 512)

    # Global pooling
    net = tf.reduce_max(self.points[4], axis=1, keepdims=False)
    # print(f'net after pooling: {net.shape}') # net after pooling: (32, 512)

    # Dense layers to reach the latent space
    fc_layer1 = fully_connected_v2(256, bn=True, bn_decay=self.bn_decay)
    net = fc_layer1(net)
    # TODO : Add tf_util.dropout(inputs, is_training=True, scope='drop1', keep_prob=0.5, noise_shape=None)
    fc_layer2 = fully_connected_v2(256, bn=True, bn_decay=self.bn_decay)
    net = fc_layer2(net)
    # TODO : Add tf_util.dropout(inputs, is_training=True, scope='dp2' keep_prob=0.5, noise_shape=None)
    fc_layer3 = fully_connected_v2(self.latent_dim, activation_fn=None) # Final layer for encoding to latent_dim
    net = fc_layer3(net)
    # print(f'final net: {net.shape}') # (32, latent_dim [128])

    return net

class PointNet2Decoder(Model):
  def __init__(self, encoder, bn_decay=None, **kwargs):
    super(PointNet2Decoder, self).__init__(**kwargs)
    self.encoder = encoder
    self.bn_decay = bn_decay

  def print_stats(self, tensor, name):
    print(f"{name}: min={tf.reduce_min(tensor)}, max={tf.reduce_max(tensor)}")

  def call(self, inputs, **kwargs):
    # Decoder
    # Feature Propagation layers
    self.encoder.points[3] = pointnet_fp_module(self.encoder.xyz[3], self.encoder.xyz[4], self.encoder.points[3], self.encoder.points[4], [256,256], True, self.bn_decay, scope='fa_layer1')
    # print(f'l3_points: {self.encoder.points[3].shape}') # l3_points: (32, 256, 256
    #self.print_stats(self.encoder.points[3], 'l3_points')
    #self.encoder.points[3] = BatchNormalization()(self.encoder.points[3])
    self.encoder.points[2] = pointnet_fp_module(self.encoder.xyz[2], self.encoder.xyz[3], self.encoder.points[2], self.encoder.points[3], [256,256], True, self.bn_decay, scope='fa_layer2')
    # print(f'l2_points: {self.encoder.points[2].shape}') # l2_points: (32, 512, 256)
    #self.print_stats(self.encoder.points[2], 'l2_points')
    self.encoder.points[1] = pointnet_fp_module(self.encoder.xyz[1], self.encoder.xyz[2], self.encoder.points[1], self.encoder.points[2], [256,128], True, self.bn_decay, scope='fa_layer3')
    # print(f'l1_points: {self.encoder.points[1].shape}') # l1_points: (32, 1024, 128)
    #self.print_stats(self.encoder.points[1], 'l1_points')
    self.encoder.points[0] = pointnet_fp_module(self.encoder.xyz[0], self.encoder.xyz[1], self.encoder.points[0], self.encoder.points[1], [128,128,128], True, self.bn_decay, scope='fa_layer4')
    # print(f'l0_points: {self.encoder.points[0].shape}') # l0_points: (32, 1024, 128)
    #self.print_stats(self.encoder.points[0], 'l0_points')

    l0_points_reshaped = tf.reshape(self.encoder.points[0], [BATCH_SIZE * self.encoder.num_point, 128]) # Reshape to 2D for applying the dense layer
    # print(f'l0_points_reshaped: {l0_points_reshaped.shape}')
    intermediate_features = tf.keras.layers.Dense(64, activation='relu')(l0_points_reshaped)
    # print(f'intermediate_features1: {intermediate_features.shape}')
    intermediate_features = tf.keras.layers.Dense(32, activation='relu')(intermediate_features)
    # print(f'intermediate_features2: {intermediate_features.shape}')
    coordinates = tf.keras.layers.Dense(3, activation='tanh')(intermediate_features)
    # print(f'coordinates1: {coordinates.shape}')
    coordinates = tf.reshape(coordinates, [BATCH_SIZE, self.encoder.num_point, 3])
    # print(f'coordinates2: {coordinates.shape}')

    return coordinates

from tensorflow.keras.optimizers import Adam
from CDLoss import ChamferDistanceLoss

class PointNet2AE(Model):
  def __init__(self, input_shape, latent_dim):
    super(PointNet2AE, self).__init__()
    self.encoder_model = PointNet2Encoder(input_shape, 128, 0.99)
    self.decoder_model = PointNet2Decoder(self.encoder_model, 0.99)

  def call(self, inputs):
    encoded_tensor = self.encoder_model(inputs)
    decoded_tensor = self.decoder_model(encoded_tensor)
    return decoded_tensor

if __name__ == "__main__":
  ae_model = PointNet2AE(point_clouds.shape[1:], 128)
  ae_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=ChamferDistanceLoss())

  # Train the model
  history = ae_model.fit(point_clouds, point_clouds, epochs=EPOCHS, batch_size=BATCH_SIZE)