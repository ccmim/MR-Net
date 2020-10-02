# MR-Net
Code for the paper "Learned Generative Shape Reconstruction from Sparse and Incomplete Point Clouds", which is a deep learning network to reconstruct 3D cardiac mesh from stacked 2D contours (point cloud).
## Contents
- [Abstract](## Abstract:)
- Network
- Repo Contents
- Package dependencies
- Training
- Testing
- Demo
- Citation

## Abstract:
Shape reconstruction from sparse point clouds/images is challenging and a relevant task for a variety of applications in computer vision and medical image analysis (e.g. surgical navigation, cardiac motion analysis, robot grasping and object manipulation). A subset of such methods, viz. 3D shape reconstruction from 2D contours, is especially relevant in computer-aided diagnosis and interventions involving meshes derived from multiple 2D image slices, views or projections. We propose a deep learning architecture, MR-Net, that tackles this problem, and enables accurate 3D mesh reconstruction in real-time in spite of missing data, and with sparse annotations. Using 3D cardiac shape reconstruction from 2D contours defined on short-axis cardiac magnetic resonance image slices as an exemplar, we demonstrate that our approach consistently outperforms state-of-the-art techniques for reconstruction from unstructured point clouds, with which the reconstructed 3D cardiac meshes is less than 3 mm of point to point distance to the ground-truth. We evaluate the robustness of the proposed approach to incomplete data and contours from automatic segmentation, and demonstrate its ability to reconstruct high-quality biventricular cardiac anatomies from few 2D contours. MR-Net is generic and could be used in other imaged organs and more general 3D objects, and this framework fills a blank in mesh reconstruction from point cloud of contours.
## Network:
The kernal idea is to use dep learning network to mimic the process of deforming the template mesh under the guidence of contours.
![image](https://github.com/XiangChen1994/MR-Net/blob/main/fig/MRNet.png)

## Repo Contents:
This code is based on [Pixel2mesh](https://github.com/nywang16/Pixel2Mesh), where the GCN block and the mesh loss are mainly from it.

## Package dependencies:
This repository is based on Python2.7, Tensorflow and Tensorlayer.
The version of the main packages is as follows,
- Tensorflow==1.7.0
- tflearn

## Training:
Use the following command to train the MR-Net.
> CUDA_VISIBLE_DEVICES=0 python train.py

## Testing:
Use the following command to test the MR-Net. Chamfer Distance (CD), Earth Mover Distance (EMD), Hausdorff Distance (HD) and Point cloud to point cloud (PC-to-PC) error are evaluated in this paper.
> CUDA_VISIBLE_DEVICES=0 python test.py

## Demo:
To reconstruct 3D cardiac mesh with pretrained model from contours.
> CUDA_VISIBLE_DEVICES=0 python demo.py

## Citation:
to do.
