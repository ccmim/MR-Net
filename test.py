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

# Modifications: revise reading and printing of testing
# Modifications copyright (C) 2013 <Xiang Chen>
#


import os, sys
import tensorflow as tf
from mrnet.models import GCN
from mrnet.fetcher import *
sys.path.append('external')
from tf_approxmatch import approx_match, match_cost
from mrnet.chamfer import nn_distance
import glob
from scipy.spatial.distance import directed_hausdorff

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'Data/Shapes/train', 'Data folder.')
flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')
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
    #'center': tf.placeholder(tf.float32, shape=(1, 3)), #center of normalisation
    #'radius': tf.placeholder(tf.float32, shape=(1,1)), #radius of normalisation
    'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks-1)] # helper for graph unpooling
}
model = GCN(placeholders, logging=True)

# Construct feed dictionary
def construct_feed_dict(pkl, placeholders):
	coord = pkl[0]

	pool_idx = pkl[5][:2]#
	faces = pkl[7][:3]
	
	lape_idx = pkl[8][:3] #7

	edges = pkl[5]#[]

	feed_dict = dict()
	feed_dict.update({placeholders['features']: coord})
	feed_dict.update({placeholders['edges'][i]: edges[i] for i in range(len(edges))})
	feed_dict.update({placeholders['faces'][i]: faces[i] for i in range(len(faces))})
	feed_dict.update({placeholders['pool_idx'][i]: pool_idx[i] for i in range(len(pool_idx))})
	feed_dict.update({placeholders['lape_idx'][i]: lape_idx[i] for i in range(len(lape_idx))})
	feed_dict.update({placeholders['support1'][i]: pkl[1][i] for i in range(len(pkl[1]))})
	feed_dict.update({placeholders['support2'][i]: pkl[2][i] for i in range(len(pkl[2]))})
	feed_dict.update({placeholders['support3'][i]: pkl[3][i] for i in range(len(pkl[3]))})
	return feed_dict

def f_score(label, predict, dist_label, dist_pred, threshold):
	num_label = label.shape[0]
	num_predict = predict.shape[0]

	f_scores = []
	for i in range(len(threshold)):
		num = len(np.where(dist_label <= threshold[i])[0])
		recall = 100.0 * num / num_label
		num = len(np.where(dist_pred <= threshold[i])[0])
		precision = 100.0 * num / num_predict

		f_scores.append((2*precision*recall)/(precision+recall+1e-8))
	return np.array(f_scores)

def write_points_in_vtp(points, outfile='points.vtp', color=None):
    """
    Method that writes a vtp file containing the given points. It can be used for any set of
    3D points. Useful to visualize control points together with mesh points in the same window.

    :param numpy.ndarray points: coordinates of the points. The shape has to be (n_points, 3).
    :param string outfile: name of the output file. The extension has to be .vtp. Default is 'points.vtp'.
    :param tuple color: tuple defining the RGB color to assign to all the points. Default is
        blue: (0, 0, 255).
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

def save_mesh(vert,face,path,id):
    pc_path = path.replace('.vtk', str(id)+'.vtp')
    write_points_in_vtp(vert,pc_path)
    vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
    mesh = np.vstack((vert, face))
    pred_path = path.replace('.vtk', str(id)+'.obj')
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')
    print ('Saved to', pred_path)
data_name = glob.glob(os.path.join(FLAGS.data_dir, '*.vtk'))   
# Load data
data = DataFetcher_test_incomplete(data_name)#DataFetcher_test
data.setDaemon(True) 
data.start()
train_number = data.number

# Initialize session
# xyz1:dataset_points * 3, xyz2:query_points * 3
xyz1=tf.placeholder(tf.float32,shape=(None, 3))
xyz2=tf.placeholder(tf.float32,shape=(None, 3))
# chamfer distance
dist1,idx1,dist2,idx2 = nn_distance(xyz1, xyz2)
# earth mover distance, notice that emd_dist return the sum of all distance
xyz11 = tf.reshape(xyz1, [1, 1578,3], name='label')
xyz12 = tf.reshape(xyz2, [1, 1578,3], name='predict')
match = approx_match(xyz11, xyz12)
emd_dist = match_cost(xyz11, xyz12, match)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
model.load(sess)



# Construct feed dictionary
pkl = pickle.load(open('Data/heart/cardiac_template.dat', 'rb')) #load template
feed_dict = construct_feed_dict(pkl, placeholders)


def chamfer_loss_np(A,B):    
    r=np.sum(A*A,2)
    r=np.reshape(r,[int(r.shape[0]),int(r.shape[1]),1])
    r2=np.sum(B*B,2)
    r2=np.reshape(r2,[int(r.shape[0]),int(r.shape[1]),1])
    t=(r-2*np.matmul(A, np.transpose(B,(0, 2, 1))) 
                                         + np.transpose(r2,(0, 2, 1)))
    return np.mean((np.min(t, axis=1)+np.min(t,axis=2))/2.0,axis=-1)

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


sum_f = []
sum_cd = []
sum_hd = []
sum_emd = []
input_pc = []
target_pc = []
predict_pc = []
pre_time = []
import datetime
starttime = datetime.datetime.now()
for iters in range(train_number):
	# Fetch training data
	img_inp, gt_pc, label, model_id = data.fetch()

	input_pc.append(img_inp)
	img_inp, center, radius = normalize_point_cloud(img_inp)
 
	_, center, radius= normalize_point_cloud(gt_pc)

	target_pc.append(gt_pc)
 
	feed_dict.update({placeholders['img_inp']: img_inp})
	feed_dict.update({placeholders['labels']: label})

	starttime_temp = datetime.datetime.now()
 
	# Training step
	predict = sess.run(model.output3, feed_dict=feed_dict)
	predict = predict*radius + center
	predict_pc.append(predict)
	#print(predict.shape)
 
	endtime_temp = datetime.datetime.now()
	pre_time.append((endtime_temp - starttime_temp).seconds)

	d1,i1,d2,i2,emd = sess.run([dist1,idx1,dist2,idx2, emd_dist], feed_dict={xyz1:gt_pc,xyz2:predict})
	cd = 0.5*(np.mean(d1) + np.mean(d2))
	hd = max(directed_hausdorff(predict, gt_pc)[0], directed_hausdorff(gt_pc, predict)[0])
	sum_hd.append(hd)

	f = f_score(gt_pc,predict,d1,d2,[0.0001, 0.0002])
	sum_f.append(f)
	sum_cd.append(cd) # cd is the mean of all distance
	sum_emd.append(emd[0]/1578) # 1578 points in the reconstructed mesh
	print ('f,cd,emd', f,cd,emd)
	print ('processed number', iters)


endtime = datetime.datetime.now()
log = open('record_evaluation.txt', 'a')

T=np.asanyarray(target_pc).reshape(-1,1578,3)
TS=np.asanyarray(predict_pc).reshape(-1,1578,3)
cd_patch = chamfer_loss_np(T,TS)
pc2pc_patch = pc_pc_error(T,TS)
print("C.D. loss for Outputs (mean+-std): ", np.mean(cd_patch),np.std(cd_patch))
print("hd for Outputs (mean+-std): ", np.mean(sum_hd),np.std(sum_hd))
print("pc error for Outputs (mean+-std): ", np.mean(pc2pc_patch),np.std(pc2pc_patch))
print("emdfor Outputs (mean+-std): ", np.mean(sum_emd),np.std(sum_emd))

f = np.array(sum_f).mean()
cd = np.array(sum_cd).mean() 
hd = np.array(sum_hd).mean() 
print ('f,cd,emd:', f, cd, hd)
print >> log, len(sum_f), f, cd, emd
log.close()
sess.close()
data.shutdown()


p2p_error = np.mean(np.sum(np.abs(T-TS),2),1)
print('p2p error:', np.mean(p2p_error), np.std(p2p_error))
p2p_error = np.mean(np.sqrt(np.sum((T-TS)**2,2)),1)
print('p2p error:', np.mean(p2p_error), np.std(p2p_error))
log.close()
sess.close()
data.shutdown()
face1 = np.loadtxt('Data/heart/heart_face1.obj', dtype='|S32')

pc_in = np.asanyarray(input_pc).reshape(-1,3000,3)
save_mesh(T[0],face1,'gt.vtk',1)
save_mesh(TS[0],face1,'predict.vtk',1)
save_mesh(pc_in[0],face1,'input.vtk',1)

print ('Testing Finished!')
print (endtime - starttime).seconds
print('mean inference time:',np.array(pre_time).mean())