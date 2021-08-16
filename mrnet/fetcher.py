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

# Modifications: add several new functions to read the .vtk .vtp files, with functions to define different contours
# Modifications copyright (C) 2013 <Xiang Chen>
#
import numpy as np
import cPickle as pickle
import threading
import Queue
import sys 
from skimage import io,transform
import vtk
from vtk.util.numpy_support import vtk_to_numpy 
from scipy.spatial import ConvexHull
from scipy.special import gammainc
import trimesh
import pandas as pd

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def readvtk_input(filename):
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
    np_point = resample_pcd(np_point,3000)
    return np_point#[:1578,:]

def readvtk_gt(filename):
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
    #np_point = resample_pcd(np_point,3000)
    return np_point#[:1578,:]

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
    #furthest_distance = np.amax(
    #    np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    furthest_distance = 100.00
    input = input / furthest_distance
    return input, centroid, furthest_distance

def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def readFaceInfo(obj_path):
	vert_list = np.zeros((1,3), dtype='float32') # all vertex coord   
	face_pts = np.zeros((1,3,3), dtype='float32') # 3 vertex on triangle face
	face_axis = np.zeros((1,3,3), dtype='float32') # x y z new axis on face plane
	with open(obj_path, 'r') as f:
		while(True):
			line = f.readline()
			if not line:
				break
			if line[0:2] == 'v ':
				t = line.split('v')[1] # ' 0.1 0.2 0.3'
				vertex = np.fromstring(t, sep=' ').reshape((1,3))
				vert_list = np.append(vert_list, vertex, axis=0)
			elif line[0:2] == 'f ':
				t = line.split() # ['f', '1//2', '1/2/3', '1']
				p1,p2,p3 = [int(t[i].split('/')[0]) for i in range(1,4)]

				points = np.array([vert_list[p1], vert_list[p2], vert_list[p3]])
				face_pts = np.append(face_pts, points.reshape(1,3,3), axis=0)

				###!!!!!!!!!!!!###
				v1 = vert_list[p2] - vert_list[p1]	# x axis
				v2 = vert_list[p3] - vert_list[p1]
				f_n = np.cross(v1, v2)		# z axis, face normal
				f_y = np.cross(v1, f_n)		# y axis
				new_axis = np.array([unit(v1), unit(f_y), unit(f_n)])
				face_axis = np.append(face_axis, new_axis.reshape((1,3,3)), axis=0)

	face_pts = np.delete(face_pts, 0, 0)
	face_axis = np.delete(face_axis, 0, 0)
	vert_list = np.delete(np.array(vert_list), 0, 0)
	return vert_list, face_pts, face_axis


def visualization(pc):
	rgb = np.ones(pc.shape)*255
	pc = np.hstack((pc,rgb))
	pc = pd.DataFrame(pc)
	cloud = PyntCloud(pc)
	voxelgrid_id = cloud.add_structure("voxelgrid", n_x=64, n_y=64, n_z=64)
	voxelgrid = cloud.structures[voxelgrid_id]
	my_arr = np.array(voxelgrid.get_feature_vector(mode="binary"))
	return my_arr

def generate_normal(pt_position, face_pts, face_axis):
	pt_normal = np.zeros_like(pt_position, dtype='float32')
	for points, axis in zip(face_pts, face_axis):
		f_org = points[0] # new axis system origin point
		f_n = axis[2] 
        
		face_vertex_2d = np.dot(points - f_org, axis.T)[:,:2]

		# check if a valid face	 
		n1,n2,n3 = [np.linalg.norm(face_axis[i]) for i in range(3)]
		if n1<0.99 or n2<0.99 or n3<0.99:
			continue
		# check if 3 point on one line
		t = np.sum(np.square(face_vertex_2d), 0)
		if t[0]==0 or t[1]==0:
			continue

		transform_verts = np.dot(pt_position - f_org, axis.transpose())
		vert_idx = np.where(np.abs(transform_verts[:,2]) < 6e-7)[0]

		for idx in vert_idx:
			if np.linalg.norm(pt_normal[idx]) == 0:
				p4 = transform_verts[idx][:2].reshape(1,2)
				pt_4 = np.append(face_vertex_2d, p4, axis=0)  
				hull = ConvexHull(pt_4)
				if len(hull.vertices) == 3:
					pt_normal[idx] = f_n * (-1)
	return np.hstack((pt_position, pt_normal))

class DataFetcher(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(64)

		self.pkl_list = []
		with open(file_list, 'r') as f:
			while(True):
				line = f.readline().strip()
				if not line:
					break
				self.pkl_list.append(line)
		self.index = 0
		self.number = len(self.pkl_list)
		np.random.shuffle(self.pkl_list)

	def work(self, idx):
		pkl_path = self.pkl_list[idx]
		img_path = pkl_path.replace('/Shapes/', '/Manual/')
		obj_path = pkl_path.replace(".vtk",".obj")
		vertices, face_pts, face_axis = readFaceInfo(obj_path)
		label = generate_normal(vertices, face_pts, face_axis)
   
		#pc
		img = readvtk_input(img_path) 
		img, _, _ = normalize_point_cloud(img) 

		return img, label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()


def readFaceInfo_r(obj_path,R):
	vert_list = np.zeros((1,3), dtype='float32') # all vertex coord   
	face_pts = np.zeros((1,3,3), dtype='float32') # 3 vertex on triangle face
	face_axis = np.zeros((1,3,3), dtype='float32') # x y z new axis on face plane
	with open(obj_path, 'r') as f:
		while(True):
			line = f.readline()
			if not line:
				break
			if line[0:2] == 'v ':
				t = line.split('v')[1] # ' 0.1 0.2 0.3'
				vertex = np.fromstring(t, sep=' ').reshape((1,3))
				vertex = np.dot(R, vertex.T).T[:,:3]
				vert_list = np.append(vert_list, vertex, axis=0)
        
			elif line[0:2] == 'f ':
				t = line.split() # ['f', '1//2', '1/2/3', '1']
				p1,p2,p3 = [int(t[i].split('/')[0]) for i in range(1,4)]

				points = np.array([vert_list[p1], vert_list[p2], vert_list[p3]])
				face_pts = np.append(face_pts, points.reshape(1,3,3), axis=0)

				###!!!!!!!!!!!!###
				v1 = vert_list[p2] - vert_list[p1]	# x axis
				v2 = vert_list[p3] - vert_list[p1]
				f_n = np.cross(v1, v2)		# z axis, face normal
				f_y = np.cross(v1, f_n)		# y axis
				new_axis = np.array([unit(v1), unit(f_y), unit(f_n)])
				face_axis = np.append(face_axis, new_axis.reshape((1,3,3)), axis=0)

	face_pts = np.delete(face_pts, 0, 0)
	face_axis = np.delete(face_axis, 0, 0)
	vert_list = np.delete(np.array(vert_list), 0, 0)
	return vert_list, face_pts, face_axis
from math import pi ,sin, cos
import math
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the user specified axis by theta radians.
    """
    # convert the input to an array
    axis = np.asarray(axis)
    # Get unit vector of our axis
    axis = axis/math.sqrt(np.dot(axis, axis))
    # take the cosine of out rotation degree in radians
    a = math.cos(theta/2.0)
    # get the rest rotation matrix components
    b, c, d = -axis*math.sin(theta/2.0)
    # create squared terms
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    # create cross terms
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    # return our rotation matrix

    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

class DataFetcher_train_advanced(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher_train_advanced, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(64)

		self.pkl_list = file_list
		self.index = 0
		self.number = len(self.pkl_list)
		np.random.shuffle(self.pkl_list)

	def work(self, idx):
 
		axis = [-1+2*np.random.random(), -1+2*np.random.random(), -1+2*np.random.random()]
		theta = 2.0*np.pi*np.random.random()
		rotat_m = rotation_matrix(axis, theta)
 
 
		pkl_path = self.pkl_list[idx]
		img_path = pkl_path.replace('/Shapes/', '/Manual/')
		obj_path = pkl_path.replace(".vtk",".obj")
		vertices, face_pts, face_axis = readFaceInfo_r(obj_path,rotat_m)
		label = generate_normal(vertices, face_pts, face_axis)
   
		#pc
		mode = np.random.randint(1,5)  #mode = np.random.randint(5)  
		img = readvtk(img_path,mode)
		img, _, _ = normalize_point_cloud(img) 
		img = np.dot(rotat_m, img.T).T[:,:3]

		return img,label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

class DataFetcher_test(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher_test, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(64)

		self.pkl_list = file_list
		self.index = 0
		self.number = len(self.pkl_list)
		np.random.shuffle(self.pkl_list)

	def work(self, idx):
		pkl_path = self.pkl_list[idx]
		img_path = pkl_path.replace('/Shapes/', '/Manual/')
		obj_path = pkl_path.replace(".vtk",".obj")
		vertices, face_pts, face_axis = readFaceInfo(obj_path)
		label = generate_normal(vertices, face_pts, face_axis)
		pc_gt = readvtk_gt(pkl_path)
   
		#pc
		img = readvtk_input(img_path) 

		return img, pc_gt,label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

import copy
from collections import Counter
from sklearn.cluster import DBSCAN
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy 
from vtk.numpy_interface import dataset_adapter as dsa
from scipy.special import gammainc
def select2(points,k):
    data_slice, counter_slice = slice_conform1(points)
    cou = Counter(counter_slice)
    cou = list(cou.keys())
    cou = np.sort(cou)
    
    counter_slice[counter_slice==0]=-1
    counter_slice[counter_slice==cou[-1]]=-1
    cluster = data_slice[counter_slice <0, :]
    cluster = resample_pcd(cluster,3000)
    return cluster
def select3(points,k):
    data_slice, counter_slice = slice_conform1(points)
    cou = Counter(counter_slice)
    cou = list(cou.keys())
    cou = np.sort(cou)
    counter_slice[counter_slice==cou[0]]=-1
    counter_slice[counter_slice==cou[len(cou)//2]]=-1
    counter_slice[counter_slice==cou[-1]]=-1
    cluster = data_slice[counter_slice <0, :]
    cluster = resample_pcd(cluster,3000)
    return cluster
def slice_conform1(data):
    db = DBSCAN(eps=5, min_samples=2).fit(data)
    labels = db.labels_
    cou = Counter(labels)
    #i = 0
    counter = 0
    counters = copy.deepcopy(labels)
    repli = []
    for i in range(len(cou)):
        if i not in repli:
            for j in range(len(cou)-i-1):
                if len(data[labels == i+j+1, :])<5:
                    counters[labels == i+j+1] = counter
                    break
                point1 = data[labels == i, :][0]
                point2 = data[labels == i, :][-1]
                point3 = data[labels == i, :][-2]
                point4 = data[labels == i+j+1, :][0]
                point5 = data[labels == i+j+1, :][-1]#-2    
                point6 = data[labels == i+j+1, :][-2]#-2  
                dis = 0.5*point2area_distance(point1, point2, point4, point6) + 0.5*point2area_distance(point1, point3, point4, point5)
                
                if dis < 0.05:
                    counters[labels==i] = counter
                    counters[labels == i+j+1] = counter
                    repli.append(i+j+1)
                else: 
                    counters[labels == i] = counter
            counter = counter + 1
    return data, counters
def select_k_slice(points,k):
    data_slice, counter_slice = slice_conform1(points)
    cou = Counter(counter_slice)
    cou = list(cou.keys())
    cou = np.sort(cou)
    counter_slice[counter_slice==cou[0]]=-1
    counter_slice[counter_slice==cou[-1]]=-1
    cou = cou[1:-1]
    cou = np.random.choice(cou, k-2, replace=False, p=None)
    for i in range(len(cou)):
        counter_slice[counter_slice==cou[i]]=-1

    cluster = data_slice[counter_slice <0, :]
    cluster = resample_pcd(cluster,3000)
    return cluster

def define_area(point1, point2, point3):
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    point3 = np.asarray(point3)
    AB = np.asmatrix(point2 - point1)
    AC = np.asmatrix(point3 - point1)
    N = np.cross(AB, AC)  # 
    # Ax+By+Cz
    Ax = N[0, 0]
    By = N[0, 1]
    Cz = N[0, 2]
    D = -(Ax * point1[0] + By * point1[1] + Cz * point1[2])
    return Ax, By, Cz, D

def point2area_distance(point1, point2, point3, point4):
    """
    """
    Ax, By, Cz, D = define_area(point1, point2, point3)
    mod_d = Ax * point4[0] + By * point4[1] + Cz * point4[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    d = abs(mod_d) / mod_area
    return d


def readvtk(filename, mode):
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
    if mode==0:
        np_point = resample_pcd(np_point,3000)
    if mode==1:
        np_point = select2(np_point,mode)
    if mode==2:
        np_point = select3(np_point,mode)
    if mode == 3:
        np_point = select_k_slice(np_point,4)
    if mode == 4:
        np_point = select_k_slice(np_point,5)
    return np_point#[:3000,:]

class DataFetcher_train(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher_train, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(64)

		self.pkl_list = file_list
		self.index = 0
		self.number = len(self.pkl_list)
		np.random.shuffle(self.pkl_list)

	def work(self, idx):
		pkl_path = self.pkl_list[idx]
		img_path = pkl_path.replace('/Shapes/', '/Manual/')
		obj_path = pkl_path.replace(".vtk",".obj")
		vertices, face_pts, face_axis = readFaceInfo(obj_path)
		label = generate_normal(vertices, face_pts, face_axis)
		pc_gt = readvtk_gt(pkl_path)
   
		#pc 
		mode = np.random.randint(1,5)  #mode = np.random.randint(5)  
		img = readvtk(img_path,mode)
		img, _, _ = normalize_point_cloud(img)

		return img,label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

class DataFetcher_test_incomplete(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher_test_incomplete, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(1)

		self.pkl_list = file_list
		self.index = 0
		self.number = len(self.pkl_list)

	def work(self, idx):
		pkl_path = self.pkl_list[idx]
		img_path = pkl_path.replace('/Shapes/', '/Manual/')
		obj_path = pkl_path.replace(".vtk",".obj")
		vertices, face_pts, face_axis = readFaceInfo(obj_path)
		label = generate_normal(vertices, face_pts, face_axis)
		pc_gt = readvtk_gt(pkl_path)
   
		#pc
		mode = 0#np.random.randint(5) 0:all, 2 slices, 3slices, 4slices, 5slices
		img = readvtk(img_path,mode)

		return img, pc_gt, label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()


class DataFetcher_test_ex3(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher_test_ex3, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(1)

		self.pkl_list = file_list
		self.index = 0
		self.number = len(self.pkl_list)

	def work(self, idx):
		pkl_path = self.pkl_list[idx]
   

		patient = pkl_path.split('/')[-1].split('.')[0]
		obj_path = '/localhome/scxc/MICCAI/MICCAI2020_code_organization/data/LVRV_Shapes_Manual/Shapes/test/'+patient+'.obj'
		img_path = '/localhome/scxc/MICCAI/MICCAI2020_code_organization/data/LVRV_Shapes_Manual/Manual/test/'+patient+'.vtk'
		gt_path = '/localhome/scxc/MICCAI/MICCAI2020_code_organization/data/LVRV_Shapes_Manual/Shapes/test/'+patient+'.vtk'

		vertices, face_pts, face_axis = readFaceInfo(obj_path)
		label = generate_normal(vertices, face_pts, face_axis)
		pc_gt = readvtk_gt(gt_path) #pkl_path

   
		#pc
		mode = 0#np.random.randint(5) 0:all, 2 slices, 3slices, 4slices, 5slices
		img = readvtk(img_path,mode)

		return img, pc_gt, label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

def readvtp(filename):
    '''Internal function to read vtp mesh files
    '''

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    readerOut = reader.GetOutput()
    points = []
    for i in range(readerOut.GetNumberOfPoints()):
        point = readerOut.GetPoint(i)
        points.append(point)
    results = np.array(points)
    return results

class DataFetcher_reseg(threading.Thread):
	def __init__(self, file_list):
		super(DataFetcher_reseg, self).__init__()
		self.stopped = False
		self.queue = Queue.Queue(1)#64

		self.pkl_list = file_list
		self.index = 0
		self.number = len(self.pkl_list)

	def work(self, idx):
		pkl_path = self.pkl_list[idx]

		patient = pkl_path.split('/')[-1].split('.')[0]
		obj_path = '/localhome/scxc/MICCAI/MICCAI2020_code_organization/data/LVRV_Shapes_Manual/Shapes/test/'+patient+'.obj'
		gt_path = '/localhome/scxc/MICCAI/MICCAI2020_code_organization/data/LVRV_Shapes_Manual/Shapes/test/'+patient+'.vtk'
		vertices, face_pts, face_axis = readFaceInfo(obj_path)
		label = generate_normal(vertices, face_pts, face_axis)
		pc_gt = readvtk_gt(gt_path)

		img = readvtp(pkl_path)
		img = resample_pcd(img,3000)
		return img, pc_gt, label, pkl_path.split('/')[-1]
	
	def run(self):
		while self.index < 90000000 and not self.stopped:
			self.queue.put(self.work(self.index % self.number))
			self.index += 1
			if self.index % self.number == 0:
				np.random.shuffle(self.pkl_list)
	
	def fetch(self):
		if self.stopped:
			return None
		return self.queue.get()
	
	def shutdown(self):
		self.stopped = True
		while not self.queue.empty():
			self.queue.get()

if __name__ == '__main__':
	file_list = sys.argv[1]
	data = DataFetcher(file_list)
	data.start()

	image,point,normal,_,_ = data.fetch()
	print image.shape
	print point.shape
	print normal.shape
	data.stopped = True
