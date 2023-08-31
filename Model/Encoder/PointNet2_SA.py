import os
import sys
import tensorflow as tf
from keras import Model

BASE_DIR = os.path.abspath('')
sys.path.append(os.path.join(BASE_DIR, 'ColabAE/Utils'))

from pointnet_util import PointNetSAModule

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