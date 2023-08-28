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
from keras.layers import Dense, Lambda

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BASE_DIR = os.path.abspath('')
sys.path.append(os.path.join(BASE_DIR, 'Utils'))
sys.path.append(os.path.join(BASE_DIR, 'Loss'))

from pointnet_util import PointNetSAModule, PointNetFPModule
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

    self.fc_layer1 = fully_connected_v2(256, bn=True, bn_decay=self.bn_decay)
    self.fc_layer2 = fully_connected_v2(256, bn=True, bn_decay=self.bn_decay)
    self.fc_layer3 = fully_connected_v2(self.latent_dim, activation_fn=None) # Final layer for encoding to latent_dim

  def print_tensor(self, tensor, name):
    # Usage: Lambda(lambda x: self.print_tensor(x, 'TensorName'))(tensor)
    tf.print(f'\n{name} Value:', tensor)
    return tensor

  def call(self, inputs, **kwargs):
    self.xyz[0] = tf.reshape(inputs, [-1, self.num_point, self.point_dim])

    self.xyz[1], self.points[1], _ = self.pointnet_sa_layer1(self.xyz[0])
    print(f'xyz[1]: {self.xyz[1].shape}, points[1]: {self.points[1].shape}') # xyz[1]: (32, 1024, 3), points[1]: (32, 1024, 64)
    self.xyz[2], self.points[2], _ = self.pointnet_sa_layer2(self.xyz[1])
    print(f'xyz[2]: {self.xyz[2].shape}, points[2]: {self.points[2].shape}') # xyz[2]: (32, 512, 3), points[2]: (32, 512, 128)
    self.xyz[3], self.points[3], _ = self.pointnet_sa_layer3(self.xyz[2])
    print(f'xyz[3]: {self.xyz[3].shape}, points[3]: {self.points[3].shape}') # xyz[3]: (32, 256, 3), points[3]: (32, 256, 256)
    self.xyz[4], self.points[4], _ = self.pointnet_sa_layer4(self.xyz[3])
    print(f'xyz[4]: {self.xyz[4].shape}, points[4]: {self.points[4].shape}') # xyz[4]: (32, 128, 3), points[4]: (32, 128, 512)

    # Global pooling
    net = tf.reduce_max(self.points[4], axis=1, keepdims=False)
    # print(f'net after pooling: {net.shape}') # net after pooling: (32, 512)

    # # Dense layers to reach the latent space
    # net = self.fc_layer1(net)
    # print(f'Encoder net1: {net.shape}')
    # # TODO : Add tf_util.dropout(inputs, is_training=True, scope='drop1', keep_prob=0.5, noise_shape=None)
    # net = self.fc_layer2(net)
    # print(f'Encoder net2: {net.shape}')
    # # TODO : Add tf_util.dropout(inputs, is_training=True, scope='dp2' keep_prob=0.5, noise_shape=None)
    # net = self.fc_layer3(net)
    # print(f'Encoder net3: {net.shape}')
    # # print(f'final net: {net.shape}') # (32, latent_dim [128])

    return self.xyz[4]


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
    self.dense_layer3 = Dense(3, activation='tanh')

  def print_tensor(self, tensor, name):
    # Usage: Lambda(lambda x: self.print_tensor(x, 'TensorName'))(tensor)
    tf.print(f'\n{name} Value:', tensor)
    return tensor

  def call(self, encoder, **kwargs):
    # Decoder
    # Feature Propagation layers TODO: Add BatchNormalization layers to normalize the data?
    self.points1[3] = self.pointnet_fp_layer1(self.encoder.xyz[3], self.encoder.xyz[4], self.encoder.points[3], self.encoder.points[4])
    print(f'points1[3]: {self.points1[3].shape}') # points1[3]: (32, 256, 256)
    self.points1[2] = self.pointnet_fp_layer2(self.encoder.xyz[2], self.encoder.xyz[3], self.encoder.points[2], self.encoder.points[3])
    print(f'points1[2]: {self.points1[2].shape}') # points1[2]: (32, 512, 256)
    self.points1[1] = self.pointnet_fp_layer3(self.encoder.xyz[1], self.encoder.xyz[2], self.encoder.points[1], self.encoder.points[2])
    print(f'points1[1]: {self.points1[1].shape}') # points1[1]: (32, 1024, 128)
    self.points1[0] = self.pointnet_fp_layer4(self.encoder.xyz[0], self.encoder.xyz[1], self.encoder.points[0], self.encoder.points[1])
    print(f'points1[0]: {self.points1[0].shape}') # points1[0]: (32, 1024, 128)

    l0_points_reshaped = tf.reshape(self.points1[0], [BATCH_SIZE * self.encoder.num_point, 128]) # Reshape to 2D for applying the dense layer
    print(f'l0_points_reshaped: {l0_points_reshaped.shape}') # l0_points_reshaped: (32768, 128)
    intermediate_features = self.dense_layer1(l0_points_reshaped)
    print(f'intermediate_features1: {intermediate_features.shape}') # intermediate_features1: (32768, 64)
    intermediate_features = self.dense_layer2(intermediate_features)
    print(f'intermediate_features2: {intermediate_features.shape}') # intermediate_features2: (32768, 32)
    coordinates = self.dense_layer3(intermediate_features)
    print(f'coordinates1: {coordinates.shape}') # coordinates1: (32768, 3)
    coordinates = tf.reshape(coordinates, [BATCH_SIZE, self.encoder.num_point, 3])
    print(f'coordinates2: {coordinates.shape}') # coordinates2: (32, 1024, 3)

    return coordinates


from tensorflow.keras.optimizers import Adam
#from EMDLoss import EMDLoss
from CDLoss import ChamferDistanceLoss

sys.path.append(os.path.join(BASE_DIR, 'Model/Decoder'))
from FC import FC

class PointNet2AE(Model):
  def __init__(self, input_shape, latent_dim):
    super(PointNet2AE, self).__init__()
    self.encoder_model = PointNet2Encoder(input_shape, 128, 0.99)
    #self.decoder_model = PointNet2Decoder(self.encoder_model, 0.99)
    self.decoder_model = FC(input_shape)

  def call(self, inputs):
    encoded_tensor = self.encoder_model(inputs)
    decoded_tensor = self.decoder_model(encoded_tensor)
    return decoded_tensor


if __name__ == "__main__":
  ae_model = PointNet2AE(point_clouds.shape[1:], 128)
  ae_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=ChamferDistanceLoss())

  # Train the model
  history = ae_model.fit(point_clouds, point_clouds, epochs=EPOCHS, batch_size=BATCH_SIZE)