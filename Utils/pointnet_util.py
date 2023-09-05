""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../Ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, '../Ops/grouping'))
sys.path.append(os.path.join(BASE_DIR, '../Ops/interpolate'))
from sampling import farthest_point_sample, gather_point
from grouping import query_ball_point, group_point, knn_point
from interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
from keras.layers import Layer

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.shape[0]
    nsample = xyz.shape[1]
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.compat.v1.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.shape[1]
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(a=new_points, perm=[0,3,1,2])
        conv2d_layers = [None] * len(mlp)
        intermediate_results = [None] * (len(mlp) + 1)
        intermediate_results[0] = new_points
        for i, num_out_channel in enumerate(mlp):
            conv2d_layers[i] = tf_util.conv2d_v2(num_out_channel, [1, 1],
                                                 padding='VALID', stride=[1, 1],
                                                 bn=bn,
                                                 scope=scope + '_conv%d' % (i), bn_decay=bn_decay,
                                                 data_format=data_format)
            intermediate_results[i+1] = conv2d_layers[i](intermediate_results[i])
        new_points = intermediate_results[-1]

        if use_nchw: new_points = tf.transpose(a=new_points, perm=[0,2,3,1])
        # Pooling in Local Regions
        if pooling=='max':
            new_points = tf.reduce_max(input_tensor=new_points, axis=[2], keepdims=True, name='maxpool')
        elif pooling=='avg':
            new_points = tf.reduce_mean(input_tensor=new_points, axis=[2], keepdims=True, name='avgpool')
        elif pooling=='weighted_avg':
            with tf.compat.v1.variable_scope('weighted_avg'):
                dists = tf.norm(tensor=grouped_xyz,axis=-1,ord=2,keepdims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(input_tensor=exp_dists,axis=2,keepdims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(input_tensor=new_points, axis=2, keepdims=True)
        elif pooling=='max_and_avg':
            max_points = tf.reduce_max(input_tensor=new_points, axis=[2], keepdims=True, name='maxpool')
            avg_points = tf.reduce_mean(input_tensor=new_points, axis=[2], keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing 
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(a=new_points, perm=[0,3,1,2])
            conv2d_layers2 = [None] * len(mlp)
            intermediate_results2 = [None] * (len(mlp) + 1)
            intermediate_results2[0] = new_points
            for i, num_out_channel in enumerate(mlp):
                conv2d_layers2[i] = tf_util.conv2d_v2(num_out_channel, [1, 1],
                                                    padding='VALID', stride=[1, 1],
                                                    bn=bn,
                                                    scope=scope+'_conv%d' % (i), bn_decay=bn_decay,
                                                    data_format=data_format)
                intermediate_results2[i + 1] = conv2d_layers2[i](intermediate_results2[i])
            new_points = intermediate_results2[-1]
            if use_nchw: new_points = tf.transpose(a=new_points, perm=[0,2,3,1])

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


''' PointNet Set Abstraction (SA) Module '''
class PointNetSAModule(Layer):
    '''
        Input:
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
    '''
    def __init__(self, npoint, radius, nsample, mlp, scope, bn_decay=None, mlp2=None, group_all=False,  bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False, **kwargs):
        super(PointNetSAModule, self).__init__(**kwargs)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.scope = scope
        self.mlp2 = mlp2
        self.group_all = group_all
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw
        self.conv2d_layers = []
        self.conv2d_layers2 = []

        data_format = 'NCHW' if self.use_nchw else 'NHWC'
        for i, num_out_channel in enumerate(self.mlp):
            self.conv2d_layers.append(
                tf_util.conv2d_v2(num_out_channel, [1, 1],
                                  padding='VALID', stride=[1, 1], bn=self.bn,
                                  scope=self.scope + '_conv%d' % (i), bn_decay=self.bn_decay,
                                  data_format=data_format)
            )
        if self.mlp2:
            for i, num_out_channel in enumerate(self.mlp2):
                self.conv2d_layers2.append(
                    tf_util.conv2d_v2(num_out_channel, [1, 1],
                                      padding='VALID', stride=[1, 1], bn=self.bn,
                                      scope=self.scope + '_conv_post%d' % (i), bn_decay=self.bn_decay,
                                      data_format=data_format)
                )

    def call(self, xyz, points, **kwargs):
        '''
            Input:
                xyz: (batch_size, ndataset, 3) TF tensor
            Return:
                new_xyz: (batch_size, npoint, 3) TF tensor
                new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
                idx: (batch_size, npoint, nsample) int32 -- indices for local regions
        '''
        # Sample and Grouping
        if self.group_all:
            self.nsample = xyz.shape[1]
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.knn, self.use_xyz)

        # Point Feature Embedding
        new_points = tf.transpose(new_points, perm=[0, 3, 1, 2]) if self.use_nchw else new_points
        intermediate_results = [new_points]
        for layer in self.conv2d_layers:
            intermediate_results.append(layer(intermediate_results[-1]))
        new_points = intermediate_results[-1]
        new_points = tf.transpose(new_points, perm=[0, 2, 3, 1]) if self.use_nchw else new_points

        # Pooling in Local Regions
        if self.pooling=='max':
            new_points = tf.reduce_max(input_tensor=new_points, axis=[2], keepdims=True, name='maxpool')
        elif self.pooling=='avg':
            new_points = tf.reduce_mean(input_tensor=new_points, axis=[2], keepdims=True, name='avgpool')
        elif self.pooling=='weighted_avg':
            dists = tf.norm(tensor=grouped_xyz,axis=-1,ord=2,keepdims=True)
            exp_dists = tf.exp(-dists * 5)
            weights = exp_dists/tf.reduce_sum(input_tensor=exp_dists,axis=2,keepdims=True) # (batch_size, npoint, nsample, 1)
            new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
            new_points = tf.reduce_sum(input_tensor=new_points, axis=2, keepdims=True)
        elif self.pooling=='max_and_avg':
            max_points = tf.reduce_max(input_tensor=new_points, axis=[2], keepdims=True, name='maxpool')
            avg_points = tf.reduce_mean(input_tensor=new_points, axis=[2], keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if self.mlp2:
            new_points = tf.transpose(new_points, perm=[0, 3, 1, 2]) if self.use_nchw else new_points
            intermediate_results2 = [new_points]
            for layer in self.conv2d_layers2:
                intermediate_results2.append(layer(intermediate_results2[-1]))
            new_points = intermediate_results2[-1]
            new_points = tf.transpose(new_points, perm=[0, 2, 3, 1]) if self.use_nchw else new_points

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.compat.v1.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(a=grouped_points, perm=[0,3,1,2])
            for j,num_out_channel in enumerate(mlp_list[i]):
                conv2d_layer = tf_util.conv2d_v2(num_out_channel, [1, 1],
                                                padding='VALID', stride=[1, 1], bn=bn,
                                                scope=scope+'_conv%d_%d'%(i,j), bn_decay=bn_decay)
                grouped_points = conv2d_layer(grouped_points)
            if use_nchw: grouped_points = tf.transpose(a=grouped_points, perm=[0,2,3,1])
            new_points = tf.reduce_max(input_tensor=grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

''' PointNet Multi-Scale Grouping Set Abstraction (SA) Module '''
class PointNetSAModuleMSG(Layer):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
    '''
    def __init__(self, npoint, radius_list, nsample_list, mlp_list, bn_decay, scope, bn=True, use_xyz=True, use_nchw=False, **kwargs):
        super(PointNetSAModuleMSG, self).__init__(**kwargs)
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp_list = mlp_list
        self.scope = scope
        self.bn_decay = bn_decay
        self.bn = bn
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw

        self.conv2d_layers = []
        for i in range(len(self.radius_list)):
            self.conv2d_layers.append([])
            for j, num_out_channel in enumerate(self.mlp_list[i]):
                self.conv2d_layers[i].append(
                    tf_util.conv2d_v2(num_out_channel, [1, 1],
                                      padding='VALID', stride=[1, 1], bn=self.bn,
                                      scope=self.scope + '_conv%d_%d' % (i, j), bn_decay=self.bn_decay)
                )

    '''
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, sum_k{mlp[k][-1]}) TF tensor
    '''
    def call(self, xyz, points, **kwargs):
        new_xyz = gather_point(xyz, farthest_point_sample(self.npoint, xyz))
        new_points_list = []
        for i in range(len(self.radius_list)):
            radius = self.radius_list[i]
            nsample = self.nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if self.use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if self.use_nchw: grouped_points = tf.transpose(a=grouped_points, perm=[0,3,1,2])
            for j, num_out_channel in enumerate(self.mlp_list[i]):
                grouped_points = self.conv2d_layers[i][j](grouped_points)
            if self.use_nchw: grouped_points = tf.transpose(a=grouped_points, perm=[0,2,3,1])
            new_points = tf.reduce_max(input_tensor=grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            points1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.compat.v1.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum(input_tensor=(1.0/dist),axis=2,keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points

        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            conv2d_layer = tf_util.conv2d_v2(num_out_channel, [1, 1],
                                             padding='VALID', stride=[1, 1], bn=bn,
                                             scope=scope+'_conv_post_%d'%(i), bn_decay=bn_decay)
            new_points1 = conv2d_layer(new_points1)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]

        return new_points1

''' PointNet Feature Propogation (FP) Module '''
class PointNetFPModule(Layer):
    '''
        Input:
            mlp: list of int32 -- output size for MLP on each point
    '''
    def __init__(self, mlp, scope, bn_decay, bn=True, **kwargs):
        super(PointNetFPModule, self).__init__(**kwargs)
        self.mlp = mlp
        self.bn_decay = bn_decay
        self.scope = scope
        self.bn = bn
        self.conv2d_layers = []

        for i, num_out_channel in enumerate(self.mlp):
            self.conv2d_layers.append(
                tf_util.conv2d_v2(num_out_channel, [1, 1],
                                 padding='VALID', stride=[1, 1], bn=self.bn,
                                 scope=self.scope+'_conv%d'%(i), bn_decay=self.bn_decay)
            )

    def call(self, xyz1, xyz2, points1, points2, **kwargs):
        '''
            Input:
                xyz1: (batch_size, ndataset1, 3) TF tensor
                xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
                points1: (batch_size, ndataset1, nchannel1) TF tensor
                points2: (batch_size, ndataset2, nchannel2) TF tensor
            Return:
                new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
            '''
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum(input_tensor=(1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        # Point Feature Embedding
        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)

        intermediate_results = [new_points1]
        for layer in self.conv2d_layers:
            print(f'{self.scope} intermediate_results[-1]: {intermediate_results[-1].shape}')
            intermediate_results.append(layer(intermediate_results[-1]))

        return tf.squeeze(intermediate_results[-1], [2])  # (batch_size, npoints, mlp2[-1])

