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

# Modifications: mesh/PC data reading and saving
# Modifications copyright (C) 2013 <Xiang Chen>
#
import tensorflow as tf
import pickle
from skimage import io,transform
from mrnet.api import GCN
from mrnet.utils import *

import vtk
from vtk.util.numpy_support import vtk_to_numpy 
from scipy.spatial import ConvexHull
from scipy.special import gammainc
import trimesh
from mrnet.fetcher import *


# Set random seed
seed = 1024
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('test_file', 'test/1000739_ED.vtk', 'Testfile dir.')
flags.DEFINE_float('learning_rate', 0., 'Initial learning rate.')
flags.DEFINE_integer('hidden', 256, 'Number of units in  hidden layer.')
flags.DEFINE_integer('feat_dim', 963, 'Number of units in perceptual feature layer.')
flags.DEFINE_integer('coord_dim', 3, 'Number of units in output layer.') 
flags.DEFINE_float('weight_decay', 5e-6, 'Weight decay for L2 loss.')

# Define placeholders(dict) and model
num_blocks = 3
num_supports = 2
placeholders = {
    'features': tf.placeholder(tf.float32, shape=(None, 3)), # initial 3D coordinates
    'img_inp': tf.placeholder(tf.float32, shape=(3000,3)), # input image to network
    'labels': tf.placeholder(tf.float32, shape=(None, 6)), # ground truth (point cloud with vertex normal)
    'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the first block
    'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the second block
    'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # graph structure in the third block
    'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)], # helper for face loss (not used)
    'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)], # helper for normal loss
    'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)], # helper for laplacian regularization
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] # helper for graph unpooling
}
model = GCN(placeholders, logging=True)

def write_points_in_vtp(points, outfile='points.vtp', color=None):
    """
    Method that writes a vtp file containing the given points. It can be used for any set of
    3D points. Useful to visualize control points together with mesh points in the same window.

    :param numpy.ndarray points: coordinates of the points. The shape has to be (n_points, 3).
    :param string outfile: name of the output file. The extension has to be .vtp. Default is 'points.vtp'.
    :param tuple color: tuple defining the RGB color to assign to all the points. Default is
        blue: (0, 0, 255).

    :Example:

    >>> import pygem.utils as ut
    >>> import numpy as np

    >>> ctrl_points = np.arange(9).reshape(3, 3)
    >>> ut.write_points_in_vtp(ctrl_points, 'example_points.vtp', color=(255, 0, 0))
    """
    if color is None:
        color = (255, 255, 255)
    # setup points and vertices
    Points = vtk.vtkPoints()
    Vertices = vtk.vtkCellArray()
    
    Colors = vtk.vtkUnsignedCharArray()
    Colors.SetNumberOfComponents(3)
    Colors.SetName("Colors")
    
    for i in range(points.shape[0]): 
        ind = Points.InsertNextPoint(points[i][0], points[i][1], points[i][2])
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(ind)
        Colors.InsertNextTuple3(color[0], color[1], color[2])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(Points)
    polydata.SetVerts(Vertices)
    polydata.GetPointData().SetScalars(Colors)
    polydata.Modified()
    if vtk.VTK_MAJOR_VERSION <= 5:
        polydata.Update()
     
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outfile)
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(polydata)
    else:
        writer.SetInputData(polydata)
    writer.Write() 

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]
def readvtkgt(filename):
    # load a vtk file as input 
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename) 
    reader.ReadAllVectorsOn()
    reader.Update() 
	#Grab a scalar from the vtk file 
    data = reader.GetOutput()
    cells = data.GetPolys()
    triangles = cells.GetData()
    points = data.GetPoints()
    np_point = vtk_to_numpy(points.GetData()) 
    return np_point
    
def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = 100.00
    input = input / furthest_distance
    return input, centroid, furthest_distance

# Load data, initialize session
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model.load(sess)

# Runing the demo
pkl = pickle.load(open('Data/heart/cardiac_template.dat', 'rb')) #init1.dat info_ellipsoid.dat
feed_dict = construct_feed_dict(pkl, placeholders)


mode = 0 #0 full,  1 2 slices, 2-4  3-5 slices.
img_inp = readvtk(FLAGS.test_file,mode)
img_inp = resample_pcd(img_inp,3000)

write_points_in_vtp(img_inp,'demo/input.vtp')
img_inp, center, radius = normalize_point_cloud(img_inp)
def save_mesh(vert,face,path,id):
    pc_path = path.replace('.vtk', str(id)+'.vtp')
    write_points_in_vtp(vert,pc_path)
    vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
    mesh = np.vstack((vert, face))
    pred_path = path.replace('.vtk', str(id)+'.obj')
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')
    print ('Saved to', pred_path)

feed_dict.update({placeholders['img_inp']: img_inp})
feed_dict.update({placeholders['labels']: np.zeros([10,6])})

vert1,vert2, vert3 = sess.run([model.output1, model.output2,model.output3], feed_dict=feed_dict)#output3
vert1 = (vert1 * radius) + center
vert2 = (vert2 * radius) + center
vert3 = (vert3 * radius) + center

face1 = np.loadtxt('Data/heart/heart_face1.obj', dtype='|S32')
face2 = np.loadtxt('Data/heart/heart_face2.obj', dtype='|S32')
face3 = np.loadtxt('Data/heart/heart_face3.obj', dtype='|S32')

test_file = 'demo/test.vtk'
save_mesh(vert1,face1,test_file,1)
save_mesh(vert2,face2,test_file,2)
save_mesh(vert3,face3,test_file,3)


