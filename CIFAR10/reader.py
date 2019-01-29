# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import os
from PIL import Image
import numpy as np

IMAGE_DEPTH = 32
IMAGE_WIDTH = 32
IMAGE_CHANNEL = 3
IMAGE_SIZE = IMAGE_DEPTH * IMAGE_WIDTH * IMAGE_CHANNEL

def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def read_images(images,labels,batchsize,startpoint):
	return_val = {}
	return_images = []
	return_labels = []
	for batch_num in range(batchsize):
		return_single_images = []
		return_single_label = []
		try:
			image = images[batch_num+startpoint]
			label = labels[batch_num+startpoint]
			for item in image:
				return_single_images.append(float(item)/255)
			## 这个地方可以优化，用现成的python函数去构造
			for item in range(10):
				if item != label:
					return_single_label.append(0)
				else:
					return_single_label.append(1)
		except:
			print("out of input images index")
			return -1
		return_labels.append(return_single_label)
		return_images.append(return_single_images)
	return_val['data'] = return_images
	return_val['label'] = return_labels
	return return_val

def read_one_image(images,labels,filename):
	image = images[0]
	label = labels[0]
	# print len(image)
	# print label
	r_image = image[0:1024]
	g_image = image[1024:2048]
	b_image = image[2048:3072]
	return_val = []
	r_image_float = []
	g_image_float = []
	b_image_float = []
	for item in r_image:
		r_image_float.append(float(item)/255)
		return_val.append(float(item)/255)
	for item in g_image:
		g_image_float.append(float(item)/255)
		return_val.append(float(item)/255)
	for item in b_image:
		b_image_float.append(float(item)/255)
		return_val.append(float(item)/255)
	img = Image.new("RGB",(IMAGE_WIDTH,IMAGE_DEPTH))
	for x in range(IMAGE_WIDTH):
		for y in range(IMAGE_DEPTH):
			r_pix = r_image[x*IMAGE_WIDTH+y]
			g_pix = g_image[x*IMAGE_WIDTH+y]
			b_pix = b_image[x*IMAGE_WIDTH+y]
			img.putpixel((y,x),(r_pix,g_pix,b_pix))
	img.save(filename)
	print(r_image_float)
	#return np.array([r_image_float,g_image_float,b_image_float])
	return np.array([return_val])

def read_meta_data(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

if __name__ == "__main__":
	arg_parser = ArgumentParser(description='Parse args')
	arg_parser.add_argument('-d', "--dataset",
                            help='Specify the input dataset for reading',
                            dest='dataset')
	args = arg_parser.parse_args()
	print("The input dataset is: {}".format(args.dataset))
	Train_files = []
	for datafile_num in range(1,6):
		Train_file_name = "{0}{1}".format(os.path.join(args.dataset,"data_batch_"),str(datafile_num))
		Train_files.append(Train_file_name)
		print Train_file_name
	meta_data = read_meta_data(os.path.join(args.dataset,"batches.meta"))
	label_names = meta_data['label_names']
	print(label_names)
	for Train_file in Train_files:
		train_data_dict = unpickle(Train_file)
		# print(train_data_dict['data'])
		# print(len(train_data_dict['data']))
		# print(len(train_data_dict['data'][0]))
		# print(train_data_dict['labels'])
		# print(len(train_data_dict['labels']))
		# print(train_data_dict['batch_label'])
		# print(train_data_dict['filenames'])
		#print(train_data_dict)
		input_data = read_one_image(train_data_dict['data'],train_data_dict['labels'],"test.png")
		print input_data.shape
		break

