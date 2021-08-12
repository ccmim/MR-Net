#  Copyright (C) 2019 Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, Yu-Gang Jiang, Fudan University
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications: revise the GCN network
# Modifications copyright (C) 2013 <Xiang Chen>
#
from __future__ import division
import tflearn
from layers import *
from pointnet_util import *

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.output1 = None
        self.output2 = None
        self.output3 = None
        self.output1_2 = None
        self.output2_2 = None

        self.loss = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        #with tf.device('/gpu:0'):
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential resnet model
        eltwise = [3,5,7,9,11,13, 19,21,23,25,27,29, 35,37,39,41,43,45]
        concat = [15, 31]
        self.activations.append(self.inputs)
        for idx,layer in enumerate(self.layers):
            hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            if idx in concat:
                hidden = tf.concat([hidden, self.activations[-2]], 1)
            self.activations.append(hidden)

        self.output1 = self.activations[15]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=1)
        self.output1_2 = unpool_layer(self.output1)

        self.output2 = self.activations[31]
        unpool_layer = GraphPooling(placeholders=self.placeholders, pool_id=2)
        self.output2_2 = unpool_layer(self.output2)

        self.output3 = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "Data/checkpoint/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "Data/checkpoint/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class GCN(Model):
    def __init__(self, placeholders, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        #self.radius = placeholders['radius']
        #self.center = placeholders['center']
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        self.loss += 0.1* mesh_loss(self.output1, self.placeholders, 1)
        self.loss += 0.3* mesh_loss(self.output2, self.placeholders, 1)
        self.loss += 0.6* mesh_loss(self.output3, self.placeholders, 1)
        self.loss += .1*laplace_loss(self.inputs, self.output1, self.placeholders, 1)
        self.loss += laplace_loss(self.inputs, self.output2, self.placeholders, 1)
        self.loss += laplace_loss(self.inputs, self.output3, self.placeholders, 1)
        self.point_loss = pc_loss(self.output3, self.placeholders)
        # Weight decay loss
        conv_layers = range(1,15) + range(17,31) + range(33,48)
        for layer_id in conv_layers:
            for var in self.layers[layer_id].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    def _build(self):
        self.build_pc_feature() #feature extraction in PC space
        self.build_cnn_3D() #feature extraction in 3D image space
		# first project block
        self.layers.append(GraphProjection_3D(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(12):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
		# second project block
        self.layers.append(GraphProjection_3D(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim+FLAGS.hidden,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(13):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
		# third project block
        self.layers.append(GraphProjection_3D(placeholders=self.placeholders))
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim+FLAGS.hidden,
                                            output_dim=FLAGS.hidden,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        for _ in range(13):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,
                                                gcn_block_id=1,
                                                placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=int(FLAGS.hidden/2),
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=int(FLAGS.hidden/2),
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,
                                            gcn_block_id=1,
                                            placeholders=self.placeholders, logging=self.logging))
    def build_pc_feature(self):
        x=self.placeholders['img_inp']
        bn_decay = 0.99
        #x=tf.expand_dims(x, 0)
        v1 = tf.expand_dims(x, 0)
        xyz1, points1, indices1 = pointnet_sa_module(v1, None, npoint=2000, radius=0.2, nsample=32, mlp=[32, 64, 128], mlp2=None, group_all=False, is_training=FLAGS.learning_rate, bn_decay=None, scope='sa_layer1', bn=False,ibn = False, pooling='max', tnet_spec=None, knn=False, use_xyz=True)
        xyz2, points2, indices2 = pointnet_sa_module(xyz1, points1, npoint=1578, radius=0.2, nsample=32, mlp=[32, 64, 128], mlp2=None, group_all=False, is_training=FLAGS.learning_rate, bn_decay=None, scope='sa_layer2', bn=False,ibn = False, pooling='max', tnet_spec=None, knn=False, use_xyz=True)

        self.placeholders.update({'pc_feat': [x, tf.squeeze(xyz1), tf.squeeze(xyz2)]})

    def build_cnn_3D(self):
        feature=self.placeholders['img_inp']
        #turn PC to 3D volume
        X1 = feature[:, 0]
        Y1 = feature[:, 1]
        Z1 = feature[:, 2]
   
        h1 = 32.0 * Y1 + 32.0
        w1 = 32.0 * X1 + 32.0
        c1 = 32.0 * Z1 + 32.0
   
        h1 = tf.minimum(tf.maximum(h1, 0), 63)
        w1 = tf.minimum(tf.maximum(w1, 0), 63)
        c1 = tf.minimum(tf.maximum(c1, 0), 63)
   
        h1 = tf.cast(tf.floor(h1),tf.int32)
        w1 = tf.cast(tf.floor(w1),tf.int32)
        c1 = tf.cast(tf.floor(c1),tf.int32)

        update = tf.ones([feature.shape[0]])*255
        index = tf.stack([h1,w1,c1],1)
        voxel_feature = tf.scatter_nd(index, update, [64,64,64])
        one = tf.ones_like(voxel_feature)*255
        zero = tf.zeros_like(voxel_feature)
        voxel_feature = tf.where(voxel_feature <0.5, x=zero, y=one)
    
        #3D CNN
        x=tf.expand_dims(voxel_feature, 3)
        x=tf.expand_dims(x, 0)
        x = tf.tile(x,[1,1,1,1,3]) #1*64*64*64*3
        x=tflearn.layers.conv.conv_3d(x,64,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_3d(x,64,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x0=x #1*64*64*64*64
        x=tflearn.layers.conv.conv_3d(x,128,(3,3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_3d(x,128,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_3d(x,128,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x1=x #1*32*32*32*128
        x=tflearn.layers.conv.conv_3d(x,256,(3,3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_3d(x,256,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_3d(x,256,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x2=x #1*16*16*16*256
        x=tflearn.layers.conv.conv_3d(x,500,(3,3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_3d(x,500,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x=tflearn.layers.conv.conv_3d(x,500,(3,3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
        x3=x #1*8*8*8*500
        #updata image feature
        self.placeholders.update({'img_feat': [tf.squeeze(x0), tf.squeeze(x1), tf.squeeze(x2), tf.squeeze(x3)]})
        self.loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.3
