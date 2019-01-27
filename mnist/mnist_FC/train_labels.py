# -*- coding: utf-8 -*-
import struct
from PIL import Image
import tensorflow as tf
import numpy as np
import datetime
import os

class train_labels():
	images_numbers = 0
	magic_number = 0
	file_obj = None
	def __init__(self,trainDataFile): #定义构造方法
		self.file_obj = open(trainDataFile,"rb")
		self.read_prefix()
	def __del__(self):
		self.file_obj.close()
	def read_prefix(self):
		try:
			bin_buf = self.file_obj.read(4) #读取二进制数组
			magic_number = struct.unpack('>i', bin_buf) #'i'代表'integer',>指原来的数据是大端 
			print("train_labels magic_number:{}".format(magic_number[0]))
			self.set_magic_number(magic_number[0])
			bin_buf = self.file_obj.read(4) #读取二进制数组
			image_number = struct.unpack('>i', bin_buf) #'i'代表'integer',>指原来的数据是大端 
			print("train_labels image_number:{0}".format(image_number[0]))
			self.set_images_number(image_number[0])
			return None
		except:
			self.file_obj.close()
	def get_images_number(self):
		return self.images_numbers
	def get_magic_number(self):
		return self.magic_number
	def set_images_number(self,images_numbers):
		self.images_numbers = images_numbers
		return self.images_numbers
	def set_magic_number(self,magic_number):
		self.magic_number = magic_number
		return self.magic_number
	def read_one_label(self):
		try:
			bin_buf = self.file_obj.read(1) #读取二进制数组
			label_val = struct.unpack('B', bin_buf)# 'i'代表'integer',>指原来的数据是大端 
			return label_val[0]
		except:
			self.file_obj.close()
	def read_labels(self,batchsize):
		try:
			label_vals = []
			for item in range(batchsize):
				bin_buf = self.file_obj.read(1) #读取二进制数组
				label_val = struct.unpack('B', bin_buf)# 'i'代表'integer',>指原来的数据是大端
				label_vals.append(label_val[0]) 
			return label_vals
		except:
			self.file_obj.close()