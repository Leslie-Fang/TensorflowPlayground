# -*- coding: utf-8 -*-
import struct
from PIL import Image
import tensorflow as tf
import numpy as np
import datetime
import os
class inference_images():
	images_numbers = 0
	magic_number = 0
	row_number = 0 #单张图片的像素点行数
	column_number = 0 #单张图片的像素点列数
	file_obj = None
	def __init__(self,trainDataFile):	#定义构造方法
		self.file_obj = open(trainDataFile,"rb")
		self.read_prefix()
	def __del__(self):
		self.file_obj.close()
	def read_prefix(self):
		try:
			bin_buf = self.file_obj.read(4) #读取二进制数组
			magic_number = struct.unpack('>i', bin_buf)# 'i'代表'integer',>指原来的数据是大端 
			print("inference_images magic_number:{0}".format(magic_number[0]))
			self.set_magic_number(magic_number[0])
			bin_buf = self.file_obj.read(4) #读取二进制数组
			image_number = struct.unpack('>i', bin_buf)# 'i'代表'integer',>指原来的数据是大端 
			print("inference_images image_number:{0}".format(image_number[0]))
			self.set_images_number(image_number[0])
			bin_buf = self.file_obj.read(4) #读取二进制数组
			row_number = struct.unpack('>i', bin_buf)# 'i'代表'integer',>指原来的数据是大端 
			print("inference_images row_number:{}".format(row_number[0]))
			self.set_row_number(row_number[0])
			bin_buf = self.file_obj.read(4) #读取二进制数组
			column_number = struct.unpack('>i', bin_buf)# 'i'代表'integer',>指原来的数据是大端 
			print("inference_images column_number:{}".format(column_number[0]))
			self.set_column_number(column_number[0])
		except:
			self.file_obj.close()
	def set_images_number(self,images_numbers):
		self.images_numbers = images_numbers
		return self.images_numbers
	def set_magic_number(self,magic_number):
		self.magic_number = magic_number
		return self.magic_number
	def set_row_number(self,row_number):
		self.row_number = row_number
		return self.row_number
	def set_column_number(self,column_number):
		self.column_number = column_number
		return self.column_number
	def get_images_number(self):
		return self.images_numbers
	def get_magic_number(self):
		return self.magic_number
	def get_row_number(self):
		return self.row_number
	def get_column_number(self):
		return self.column_number
	def read_one_image(self,filename):
		try:
			images_pix = []
			images_pix_float = []
			for i in range(int(self.row_number)*int(self.column_number)):
				bin_buf = self.file_obj.read(1) #读取二进制数组
				pix = struct.unpack('B', bin_buf) #'i'代表'integer',>指原来的数据是大端 
				images_pix.append(pix[0])
				images_pix_float.append(pix[0]/255.0)
			img = Image.new("L",(self.column_number,self.row_number))
			for x in range(int(self.row_number)):
				for y in range(int(self.column_number)):
					img.putpixel((y,x),images_pix[x*int(self.row_number)+y])
			img.save(filename)
			return images_pix_float
		except:
			print("error")
			self.file_obj.close()
	def read_images(self,batchsize):
		try:
			images_pix_float = []
			for item in range(batchsize):
				image_pix_float = []
				for i in range(int(self.row_number)*int(self.column_number)):
					bin_buf = self.file_obj.read(1) #读取二进制数组
					pix = struct.unpack('B', bin_buf) #'i'代表'integer',>指原来的数据是大端 
					image_pix_float.append(pix[0]/255.0)
				images_pix_float.append(image_pix_float)
			return images_pix_float
		except:
			print("error")
			self.file_obj.close()