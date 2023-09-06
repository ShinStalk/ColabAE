import os
import sys
import tensorflow as tf
from keras import Model
from keras.layers import Dropout, Lambda

BASE_DIR = os.path.abspath('')
sys.path.append(os.path.join(BASE_DIR, 'Utils'))

print(BASE_DIR)
from pointnet_util import PointNetSAModuleMSG, PointNetSAModule
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

    #Branch 1
    self.pointnet_sa_layer1_br1 = PointNetSAModule(npoint=512, radius=0.2, nsample=32, mlp=[64, 64, 128], scope='sa1', bn_decay=self.bn_decay)
    self.pointnet_sa_layer2_br1 = PointNetSAModule(npoint=64, radius=0.4, nsample=64, mlp=[128, 128, 256], scope='sa2', bn_decay=self.bn_decay)
    # Branch 2
    self.pointnet_sa_layer3_br2 = PointNetSAModule(npoint=512, radius=0.4, nsample=64, mlp=[128,128,256], scope='sa3', bn_decay=self.bn_decay)
    # Branch 3
    self.pointnet_sa_layer4_br3 = PointNetSAModule(npoint=None, radius=None, nsample=None, mlp=[64, 128, 256, 512], group_all=True, scope='sa4', bn_decay=self.bn_decay)
    # Branch 4
    self.pointnet_sa_layer5_br4 = PointNetSAModule(npoint=None, radius=None, nsample=None, mlp=[256, 512, 1024], group_all=True, scope='sa5', bn_decay=self.bn_decay)

    self.fc1 = fully_connected_v2(1024)
    self.dp1 = Dropout(0.3)
    self.fc2 = fully_connected_v2(512)
    self.dp2 = Dropout(0.3)
    self.fc3 = fully_connected_v2(256)

  def call(self, inputs, **kwargs):
    self.xyz[0] = tf.reshape(inputs, [-1, self.num_point, self.point_dim])

    # Branch 1
    self.xyz[1], self.points[1], _ = self.pointnet_sa_layer1_br1(self.xyz[0], self.points[0])  # xyz[1]: (32, 512, 3), points[1]: (32, 512, 128)
    print(f'xyz[1]: {self.xyz[1].shape}, points[1]: {self.points[1].shape}')
    self.xyz[1], self.points[1], _ = self.pointnet_sa_layer2_br1(self.xyz[1], self.points[1])  # xyz[2]: (32, 64, 3), points[2]: (32, 64, 256)
    print(f'xyz[1]: {self.xyz[1].shape}, points[1]: {self.points[1].shape}')

    # Branch 2
    self.xyz[2], self.points[2], _ = self.pointnet_sa_layer3_br2(self.xyz[0], self.points[0]) # xyz[3]: (32, 512, 3), points[3]: (32, 512, 256)
    print(f'xyz[2]: {self.xyz[2].shape}, points[2]: {self.points[2].shape}')

    # Branch 1 + Branch 2
    concat12_xyz = tf.concat([self.xyz[1], self.xyz[2]], axis=1)  # concat_xyz: (32, 576(64+512), 3)
    concat12_points = tf.concat([self.points[1], self.points[2]], axis=1)  # concat_xyz: (32, 576(64+512), 256)
    print(f'concat12_xyz: {concat12_xyz.shape}, concat12_points: {concat12_points.shape}')

    # Branch 3
    self.xyz[3], self.points[3], _ = self.pointnet_sa_layer4_br3(self.xyz[0], self.points[0])  # xyz[3]: (32, 1, 3), points[3]: (32, 1, 512)
    print(f'xyz[3]: {self.xyz[3].shape}, points[3]: {self.points[3].shape}')

    # Branch 4
    self.xyz[4], self.points[4], _ = self.pointnet_sa_layer5_br4(concat12_xyz, concat12_points)  # xyz[4]: (32, 1, 3), points[4]: (32, 1, 1024)
    print(f'xyz[4]: {self.xyz[4].shape}, points[4]: {self.points[4].shape}')

    # Branch 3 + Branch 4
    concat34_xyz = tf.concat([self.xyz[3], self.xyz[4]], axis=2)  # concat_xyz: (32, 1, 3+3)
    concat34_points = tf.concat([self.points[3], self.points[4]], axis=2)  # concat_xyz: (32, 1, 512+1024)
    print(f'concat34_xyz: {concat34_xyz.shape}, concat34_points: {concat34_points.shape}')

    features = tf.squeeze(concat34_points, [1])  # points[3]: (32, 1536)
    print(f'features: {features.shape}')

    net = self.fc1(features) # fc1: (32, 1024)
    print(f'fc1 : {net.shape}')
    net = self.dp1(net)
    net = self.fc2(net) # fc2: (32, 512)
    print(f'fc2 : {net.shape}')
    net = self.dp2(net)
    net = self.fc3(net)  # fc2: (32, 256)
    print(f'fc3 : {net.shape}')

    return self.xyz[4], net

  def print_tensor(self, tensor, name):
    # Usage: Lambda(lambda x: self.print_tensor(x, 'TensorName'))(tensor)
    tf.print(f'\n{name} Value:', tensor)
    return tensor
