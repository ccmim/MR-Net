import os
import tensorflow as tf
from tensorflow.python.framework import ops
nn_distance_module=tf.load_op_library('./external/tf_nndistance_so.so')
sampling_module=tf.load_op_library('./external/tf_sampling_so.so')
grouping_module=tf.load_op_library('./external/tf_grouping_so.so')

import numpy as np


def prob_sample(inp,inpr):
    '''
input:
    batch_size * ncategory float32
    batch_size * npoints   float32
returns:
    batch_size * npoints   int32
    '''
    return sampling_module.prob_sample(inp,inpr)
ops.NoGradient('ProbSample')
# TF1.0 API requires set shape in C++
#@tf.RegisterShape('ProbSample')
#def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp,idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return sampling_module.gather_point(inp,idx)
#@tf.RegisterShape('GatherPoint')
#def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sampling_module.gather_point_grad(inp,idx,out_g),None]
def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')

def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPoint')
def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return grouping_module.selection_sort(dist, k)
ops.NoGradient('SelectionSort')
def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)
@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = xyz1.get_shape()[0].value
    n = xyz1.get_shape()[1].value
    c = xyz1.get_shape()[2].value
    m = xyz2.get_shape()[1].value
    print (b, n, c, m)
    print (xyz1, (b,1,n,c))
    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)
    print( dist, k)
    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])
    print( idx, val)
    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


# def lrelu2(x, leak=0.2, name="lrelu"):
#     with tf.variable_scope(name):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)

def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def sample(xyz, npoint):
    new_xyz = gather_point(xyz, farthest_point_sample(npoint,xyz)) # (batch_size, npoint, 3)
    return new_xyz
def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz =gather_point(xyz, farthest_point_sample(npoint, xyz)) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        if np.isscalar(radius):
            idx, pts_cnt = query_ball_point(  radius, nsample, xyz, new_xyz)
        else:
            idx_list = []
            for radius_one, xyz_one, new_xyz_one in zip(tf.unstack(radius,axis=0), tf.unstack(xyz, axis=0),tf.unstack(new_xyz, axis=0)):
                idx_one, _ = query_ball_point(radius_one, nsample, tf.expand_dims(xyz_one, axis=0), tf.expand_dims(new_xyz_one, axis=0))
                idx_list.append(idx_one)
            idx = tf.stack(idx_list, axis=0)
            idx = tf.squeeze(idx, axis=1)

    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            # new_points = tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]),grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points],axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        # new_points =  tf.concat([grouped_xyz, tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])], axis=-1)
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
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
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


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training,
                       bn_decay, scope, bn=True, ibn=False, pooling='max', tnet_spec=None, knn=False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            batch_radius: the size of each object
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec, knn, use_xyz)
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp):
            new_points = conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, ibn=ibn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay) 
        if pooling=='avg':
            new_points = tf.layers.average_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf.layers.max_pooling2d(-1 * new_points, [1, nsample], [1, 1], padding='VALID',name='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf.layers.max_pooling2d(new_points, [1,nsample], [1,1], padding='VALID', name='maxpool1')
            max_points = tf.layers.average_pooling2d(new_points, [1,nsample],[1,1], padding='VALID', name='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
            
        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, ibn=ibn,is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay) 
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope, bn=True, ibn = False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.expand_dims(new_xyz, 2)
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = conv2d(grouped_points, num_out_channel, [1,1],
                                                padding='VALID', stride=[1,1], bn=bn, ibn=ibn,is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat

 
