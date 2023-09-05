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
    self.xyz = [None, None, None, None]
    self.points = [None, None, None, None]

    self.pointnet_sa_msg_layer1 = PointNetSAModuleMSG(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128], mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]], scope='sa_msg1', bn_decay=self.bn_decay)
    self.pointnet_sa_msg_layer2 = PointNetSAModuleMSG(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128], mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]], scope='sa_msg2', bn_decay=self.bn_decay)
    self.pointnet_sa_layer3 = PointNetSAModule(npoint=None, radius=None, nsample=None, mlp=[256, 512, 1024], group_all=True, scope='sa3', bn_decay=self.bn_decay)
    self.fc1 = fully_connected_v2(512)
    self.dp1 = Dropout(0.3)
    self.fc2 = fully_connected_v2(256)
    self.dp2 = Dropout(0.3)
    self.fc3 = fully_connected_v2(128)
    #The multi - scale grouping(MSG) network(PointNet++) architecture is as follows:
    # SA_MSG(512, [0.1, 0.2, 0.4], [[32, 32, 64], [64, 64, 128], [64, 96, 128]]) →
    # SA_MSG(128, [0.2, 0.4, 0.8], [[64, 64, 128], [128, 128, 256], [128, 128, 256]]) →
    # SA_GRP_ALL([256, 512, 1024]) →
    # FC(512) → DP(0.5) → FC(256) → DP(0.5) → FC(K)

  def call(self, inputs, **kwargs):
    self.xyz[0] = tf.reshape(inputs, [-1, self.num_point, self.point_dim])

    self.xyz[1], self.points[1] = self.pointnet_sa_msg_layer1(self.xyz[0], self.points[0]) # xyz[1]: (32, 512, 3), points[1]: (32, 512, 320)
    print(f'xyz[1]: {self.xyz[1].shape}, points[1]: {self.points[1].shape}')

    self.xyz[2], self.points[2] = self.pointnet_sa_msg_layer2(self.xyz[1], self.points[1]) # xyz[2]: (32, 128, 3), points[1]: (32, 512, 320)
    print(f'xyz[2]: {self.xyz[2].shape}, points[2]: {self.points[2].shape}')

    self.xyz[3], self.points[3], _ = self.pointnet_sa_layer3(self.xyz[2], self.points[2]) # xyz[3]: (32, 1, 3), points[3]: (32, 1, 1024)
    print(f'xyz[3]: {self.xyz[3].shape}, points[3]: {self.points[3].shape}')
    #Lambda(lambda x: self.print_tensor(x, 'values points[3]'))(self.points[3])

    self.points[3] = tf.squeeze(self.points[3], [1]) # points[3]: (32, 1024)

    self.points[3] = self.fc1(self.points[3]) # points[3]: (32, 512)
    print(f'fc1 points[3]: {self.points[3].shape}')
    self.points[3] = self.dp1(self.points[3])
    self.points[3] = self.fc2(self.points[3]) # points[3]: (32, 256)
    print(f'fc2 points[3]: {self.points[3].shape}')
    self.points[3] = self.dp2(self.points[3])
    self.points[3] = self.fc3(self.points[3]) # points[3]: (32, 128)
    print(f'fc3 points[3]: {self.points[3].shape}')

    return self.xyz[3], self.points[3]

  def print_tensor(self, tensor, name):
    # Usage: Lambda(lambda x: self.print_tensor(x, 'TensorName'))(tensor)
    tf.print(f'\n{name} Value:', tensor)
    return tensor
