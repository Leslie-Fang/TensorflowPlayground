# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import os
from PIL import Image
import numpy as np
from reader import read_one_image
from reader import read_images
from PIL import Image
import tensorflow as tf
import datetime

def unpickle(file):
	import cPickle
	with open(file, 'rb') as fo:
		dict = cPickle.load(fo)
	return dict

def get_inference_data(images,labels,batchsize,startpoint):
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

if __name__ == "__main__":
	arg_parser = ArgumentParser(description='Parse args')
	arg_parser.add_argument('-d', "--dataset",
                            help='Specify the input dataset for reading',
                            dest='dataset')
	arg_parser.add_argument('-g', "--input_graph",
                            help='Specify the input graph for reading',
                            dest='pb_file',
                            default="checkPoint/graph.pb")
	args = arg_parser.parse_args()
	print("The input dataset is: {}".format(args.dataset))
	print("Begin inference!")
	base_path = os.getcwd()
	start_point = 0
	batchsize = 1
	right_count = 0
	with tf.Session() as sess:
		#pb_file = "checkPoint/graph.pb"
		#pb_file = "checkPoint/int8_graph.pb"
		with open(os.path.join(base_path,args.pb_file), 'rb') as graph:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(graph.read())
				# 获取需要进行计算的operator
			Y_,X,Ys = tf.import_graph_def(graph_def, return_elements=['Y_:0','X_:0','Ys:0'])
			inference_data_dict = unpickle(os.path.join(args.dataset,"test_batch"))
			while True:
				input_data = get_inference_data(inference_data_dict['data'],inference_data_dict['labels'],batchsize,start_point)
				if input_data == -1:
					break
				start_point = start_point + batchsize
				inference_x = np.array(input_data["data"],dtype=np.float32)
				inference_y = np.array(input_data["label"],dtype=np.float32)
				# print inference_x.shape
				# print inference_y.shape
				results = sess.run(Ys,feed_dict={X:inference_x, Y_:inference_y})
				for image_number in range(batchsize):
					maxindex  = np.argmax(results[image_number])
					true_label = np.argmax(inference_y[image_number])
					if maxindex == true_label:
						right_count = right_count + 1
			print("right_count is:{}".format(right_count))
			print("total dataset is:{}".format(10000))
			print("accuracy is:{}".format(float(right_count)/10000))
			#print train_y[0]
	# with tf.Session() as sess:
	# 	with open(os.path.join(base_path,"checkPoint/graph.pb"), 'rb') as graph:
	# 		graph_def = tf.GraphDef()
	# 		graph_def.ParseFromString(graph.read())
	# 			# 获取需要进行计算的operator
	# 		Y_,X,Ys = tf.import_graph_def(graph_def, return_elements=['Y_:0','X_:0','Ys:0'])