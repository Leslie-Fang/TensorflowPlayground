# -*- coding: utf-8 -*-
import os
from ILSVRC2014_devkit.data import load_meta_clsloc
import cv2
import numpy as np
from prepare_data import readImages
import tensorflow as tf
import datetime

def get_inference_data(batchsize,start_point):
	#总共50,000
	base_path = "/home/lesliefang/il2014_CLS_LOC/inference_data"
	# for image in os.listdir(base_path):
	# 	print(image)
	# 	print(image.split('_')[2].split(".")[0])
	# 	label_val = (int(image.split('_')[2].split(".")[0])-1)/50+1
	# 	print(label_val)
	# 	exit(1)
	return_images = []
	return_labels = []
	returen_val = {}
	readed_image_number = 0
	images = os.listdir(base_path)
	for i in range(start_point,len(images)):
		image = images[i]
		label_val = (int(image.split('_')[2].split(".")[0])-1)/50+1
		label = np.zeros((1000,), dtype=int) #总共1000个类别
		label[label_val-1] = 1

		img = cv2.imread(os.path.join(base_path,image))
		img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
		single_img1 = []
		single_img2 = []
		single_img3 = []
		for m in range(len(img)):
			single_img2 = []
			for p in range(len(img[0])):
				single_img3 = []
				for q in range(len(img[0][0])):
					niii = float(img[m][p][q])/255.0
					single_img3.append(niii)
				single_img2.append(single_img3)
			single_img1.append(single_img2)
		return_images.append(single_img1)
		return_labels.append(label)
		readed_image_number = readed_image_number + 1
		if readed_image_number >= batchsize:
			returen_val['data'] = return_images
			returen_val['label'] = return_labels
			return returen_val

batchsize = 1
start_point = 0
right_count = 0
if __name__ == "__main__":
	base_path = os.getcwd()
	pb_file = "checkPoint/graph.pb"
	with tf.Session() as sess:
		with open(os.path.join(base_path,pb_file), 'rb') as graph:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(graph.read())
			# 获取需要进行计算的operator
			Y_,X,keep_prob1,keep_prob2,Ys = tf.import_graph_def(graph_def, return_elements=['Y_:0','X_:0','keep_prob1:0','keep_prob2:0','Ys:0'])
			while True:
				input_data = get_inference_data(batchsize,start_point)
				inference_x = np.array(input_data["data"],dtype=np.float32)
				inference_y = np.array(input_data["label"],dtype=np.float32)
				results = sess.run(Ys,feed_dict={X:inference_x, Y_:inference_y,keep_prob1:0.5, keep_prob2:0.5})
				for image_number in range(batchsize):
					maxindex  = np.argmax(results[image_number])
					true_label = np.argmax(inference_y[image_number])
					print("maxindex is: {}".format(maxindex))
					print("true_label is: {}".format(true_label))
					if maxindex == true_label:
						right_count = right_count + 1
				start_point = start_point + batchsize 
				if (start_point + batchsize) > 50000:
				#if (start_point + batchsize) > 1000:
				#if True:
					print("Not enough images for next round. Finished inference!")
					print("right_count is:{}".format(right_count))
					print("total dataset is:{}".format(start_point))
					print("accuracy is:{}".format(float(right_count)/start_point))
					exit(0)
