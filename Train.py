import glob
import os
import sys

import numpy as np

NB_OF_POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Loading the point clouds data
point_cloud_files = glob.glob(os.path.join("Data/PointClouds"+str(NB_OF_POINTS), "*.npy"))
point_clouds = [np.load(pc_file) for pc_file in point_cloud_files]
point_clouds = np.stack(point_clouds)

import tensorflow as tf
from keras import Model
from keras.layers import Dense, Lambda, Dropout

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BASE_DIR = os.path.abspath('')
sys.path.append(os.path.join(BASE_DIR, 'Utils'))
sys.path.append(os.path.join(BASE_DIR, 'Loss'))
sys.path.append(os.path.join(BASE_DIR, 'Model/Decoder'))

from pointnet_util import PointNetSAModule, PointNetFPModule
from tf_util import fully_connected_v2
from FP_FC import PointNet2Decoder

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

  def print_tensor(self, tensor, name):
    # Usage: Lambda(lambda x: self.print_tensor(x, 'TensorName'))(tensor)
    tf.print(f'\n{name} Value:', tensor)
    return tensor

  def call(self, inputs, **kwargs):
    self.xyz[0] = tf.reshape(inputs, [-1, self.num_point, self.point_dim])

    self.xyz[1], self.points[1], _ = self.pointnet_sa_layer1(self.xyz[0]) # xyz[1]: (32, 1024, 3), points[1]: (32, 1024, 64)
    self.xyz[2], self.points[2], _ = self.pointnet_sa_layer2(self.xyz[1]) # xyz[2]: (32, 512, 3), points[2]: (32, 512, 128)
    self.xyz[3], self.points[3], _ = self.pointnet_sa_layer3(self.xyz[2]) # xyz[3]: (32, 256, 3), points[3]: (32, 256, 256)
    self.xyz[4], self.points[4], _ = self.pointnet_sa_layer4(self.xyz[3]) # xyz[4]: (32, 128, 3), points[4]: (32, 128, 512)

    return self.xyz[4]

#
# class PointNet2Decoder(Model):
#   def __init__(self, encoder, bn_decay=None, **kwargs):
#     super(PointNet2Decoder, self).__init__(**kwargs)
#     self.encoder = encoder
#     self.bn_decay = bn_decay
#     self.points1 = [None, None, None, None]
#     self.pointnet_fp_layer1 = PointNetFPModule([256,256], scope='fp1', bn_decay=self.bn_decay, bn=True)
#     self.pointnet_fp_layer2 = PointNetFPModule([256,256], scope='fp2', bn_decay=self.bn_decay, bn=True)
#     self.pointnet_fp_layer3 = PointNetFPModule([256,128], scope='fp3', bn_decay=self.bn_decay, bn=True)
#     self.pointnet_fp_layer4 = PointNetFPModule([128,128,128], scope='fp4', bn_decay=self.bn_decay, bn=True)
#     self.dense_layer1 = Dense(64, activation='relu')
#     self.dense_layer2 = Dense(32, activation='relu')
#     self.dense_layer3 = Dense(3, activation='tanh')
#     # self.dense_layer1 = fully_connected_v2(256, bn=True, bn_decay=self.bn_decay)
#     # self.dense_layer2 = fully_connected_v2(256, bn=True, bn_decay=self.bn_decay)
#     # self.dense_layer3 = fully_connected_v2(self.latent_dim, activation_fn=None)
#     self.dp_layer1 = Dropout(.3)
#     self.dp_layer2 = Dropout(.3)
#
#   def print_tensor(self, tensor, name):
#     # Usage: Lambda(lambda x: self.print_tensor(x, 'TensorName'))(tensor)
#     tf.print(f'\n{name} Value:', tensor)
#     return tensor
#
#   def normalize(self, tensor):
#     min_vals = tf.reduce_min(tensor, axis=-1, keepdims=True)
#     max_vals = tf.reduce_max(tensor, axis=-1, keepdims=True)
#     scaled_tensor = (tensor - min_vals) / (max_vals - min_vals)
#     # scaled_tensor = (scaled_tensor * 2) - 1 # Rescale to [-1, 1] (?)
#     return scaled_tensor
#
#   def call(self, inputs, **kwargs):
#     # Decoder
#     # Feature Propagation layers
#     self.points1[3] = self.pointnet_fp_layer1(self.encoder.xyz[3], inputs, self.encoder.points[3], self.encoder.points[4]) # points1[3]: (32, 256, 256)
#     self.points1[2] = self.pointnet_fp_layer2(self.encoder.xyz[2], self.encoder.xyz[3], self.encoder.points[2], self.points1[3]) # points1[2]: (32, 512, 256)
#     self.points1[1] = self.pointnet_fp_layer3(self.encoder.xyz[1], self.encoder.xyz[2], self.encoder.points[1], self.points1[2]) # points1[1]: (32, 1024, 128)
#     self.points1[0] = self.pointnet_fp_layer4(self.encoder.xyz[0], self.encoder.xyz[1], self.encoder.points[0], self.points1[1]) # points1[0]: (32, 1024, 128)
#
#     points1_0_shape = self.points1[0].shape
#     l0_points_reshaped = tf.reshape(self.points1[0], [points1_0_shape[0] * points1_0_shape[1], points1_0_shape[2]]) # (32768, 128)
#
#     net = self.dp_layer1(l0_points_reshaped)
#     net = self.dense_layer1(net) # (32768, 64)
#     net = self.dp_layer2(net)
#     net = self.dense_layer2(net) # (32768, 32)
#
#     # Features value normalization
#     net = self.normalize(net)
#
#     net = self.dense_layer3(net) # (32768, 3)
#     coordinates = tf.reshape(net, [BATCH_SIZE, self.encoder.num_point, 3]) # (32, 1024, 3)
#
#     return coordinates


from tensorflow.keras.optimizers import Adam
#from EMDLoss import EMDLoss
from CDLoss import ChamferDistanceLoss

sys.path.append(os.path.join(BASE_DIR, 'Model/Decoder'))
from FC import FC

class PointNet2AE(Model):
  def __init__(self, input_shape, latent_dim):
    super(PointNet2AE, self).__init__()
    self.encoder_model = PointNet2Encoder(input_shape, 128, 0.99)
    self.decoder_model = PointNet2Decoder(self.encoder_model, 0.99)
    #self.decoder_model = FC(input_shape[1])

  def call(self, inputs):
    encoded_tensor = self.encoder_model(inputs)
    decoded_tensor = self.decoder_model(encoded_tensor)
    return decoded_tensor


if __name__ == "__main__":
  ae_model = PointNet2AE(point_clouds.shape[1:], 128)
  ae_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=ChamferDistanceLoss())

  # Tensorboard
  import datetime
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  # Train the model
  history = ae_model.fit(point_clouds, point_clouds, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tensorboard_callback])

  # plot training history
  import matplotlib.pyplot as plt

  plt.plot(history.history['loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train'], loc='upper right')

  file_name = 'PN2_SA_FP_EMD_' + str(EPOCHS) + 'EP_' + '{:.0e}'.format(LEARNING_RATE) + 'LR_' + str(BATCH_SIZE) + 'BS_' + str(NB_OF_POINTS) + 'PT.h5'
  print(f'file_name: {file_name}')
  ae_model.save_weights(os.path.join('Weights/', file_name))

  # self.projection_layer1 = Dense(3, activation='linear')
  # self.projection_layer2 = Dense(3, activation='linear')
  # self.projection_layer3 = Dense(3, activation='linear')
  # self.projection_layer4 = Dense(3, activation='linear')
  #
  # normalized_features2 = self.normalize(self.points1[2])
  # normalized_features1 = self.normalize(self.points1[1])
  # normalized_features0 = self.normalize(self.points1[0])

  # Feature concatenation
  # concat_points1_3 = tf.concat([self.points1[3], self.encoder.xyz[3]], axis=-1) # (32, 256, 256+3)
  # concat_points1_2 = tf.concat([self.points1[2], self.encoder.xyz[2]], axis=-1) # (32, 512, 256+3)
  # concat_points1_1 = tf.concat([self.points1[1], self.encoder.xyz[1]], axis=-1) # (32, 1024, 128+3)
  # concat_points1_0 = tf.concat([self.points1[0], self.encoder.xyz[0]], axis=-1) # (32, 1024, 128+3)
  #
  # # Residual connections
  # projected_points1_3 = self.projection_layer1(concat_points1_3)
  # residual_points1_3 = projected_points1_3 + self.encoder.xyz[3] # (32, 256, 3)
  # projected_points1_2 = self.projection_layer1(concat_points1_2)
  # residual_points1_2 = projected_points1_2 + self.encoder.xyz[2] # (32, 512, 3)
  # projected_points1_1 = self.projection_layer1(concat_points1_1)
  # residual_points1_1 = projected_points1_1 + self.encoder.xyz[1] # (32, 1024, 3)
  # projected_points1_0 = self.projection_layer1(concat_points1_0)
  # residual_points1_0 = projected_points1_0 + self.encoder.xyz[0] # (32, 1024, 3)
  #
  # concatenated_residuals = tf.concat([residual_points1_3, residual_points1_2, residual_points1_1, residual_points1_0], axis=1) #(32, 2816, 3)
  # Lambda(lambda x: self.print_tensor(x, 'concat_points1_3'))(concat_points1_3[-4:])