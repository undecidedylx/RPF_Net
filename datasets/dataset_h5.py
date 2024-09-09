from __future__ import print_function, division
import os

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle

from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

from random import randrange

def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.9533873, 0.86847025, 0.93092921)
		std = (0.10897621, 0.21482059, 0.1544507)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord


def kernel_enhance(image):
	# plt.imshow(image)
	# plt.show()
	# 创建腐蚀和膨胀的核
	kernel = np.ones((3, 3), np.uint8)

	# 应用腐蚀操作
	erosion_enhance = cv2.erode(image, kernel, iterations=1)
	# plt.imshow(erosion_enhance)

	# plt.show()
	return erosion_enhance
def CLAHE_enhance(image):
	# plt.imshow(image)
	# plt.show()
	b, g, r = cv2.split(image)
	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
	enhance_b = clahe.apply(b)
	enhance_g = clahe.apply(g)
	enhance_r = clahe.apply(r)
	enhance_image = cv2.merge((enhance_b, enhance_g, enhance_r))
	# plt.imshow(enhance_image)
	# plt.show()

	return enhance_image
def color_enhance(image):
	# plt.imshow(image)
	# plt.show()

	alpha = 1.0  # 亮度增益
	beta = 3  # 对比度增益
	enhanced_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

	# 增强颜色饱和度
	hsv_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
	saturation_factor = 2.0  # 饱和度增益
	hsv_image[:, :, 1] = cv2.convertScaleAbs(hsv_image[:, :, 1] * saturation_factor)

	color_enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
	# plt.imshow(color_enhanced_image)
	# plt.show()

	return color_enhanced_image

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		patch_img = np.array(img)
		# print(self.file_path)
		# file_name = self.file_path.split('/')[-1].split('.')[0]
		# print(file_name)
		# enhance_color_img = color_enhance(patch_img)
		# enhance_CLAHE_img = CLAHE_enhance(patch_img)
		enhance_kernel_img = kernel_enhance(patch_img)
		# plt.imshow(patch_img)
		# plt.show()

		if self.target_patch_size is not None:
			enhance_kernel_img = enhance_kernel_img.resize(self.target_patch_size)
		enhance_kernel_img = self.roi_transforms(enhance_kernel_img).unsqueeze(0)
		# img_path = os.path.join('/remote-home/sunxinhuan/PycharmProject/data/GY_SY_mutil_slice_ccRCC_23_1009/img/0',str(file_name) + '_' + str(coord) +'.jpg')
		# # print(img_path)
		# cv2.imwrite(img_path,patch_img)
		return enhance_kernel_img, coord

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path,encoding='unicode_escape')
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




