# -*- coding: utf-8 -*-
import struct
from PIL import Image
import tensorflow as tf
import numpy as np
import datetime
import os
from inference_images import inference_images
from inference_labels import inference_labels

#train_epoch:2 accuracy:0.4951
if __name__ == "__main__":
	print("Begin inference!")
	base_path = os.getcwd()
	base_inference_path = os.path.join(base_path,"test_data")
	inference_image_path = os.path.join(base_inference_path,"t10k-images-idx3-ubyte")
	inference_label_path = os.path.join(base_inference_path,"t10k-labels-idx1-ubyte")
	inference_labels = inference_labels(inference_label_path)
	inference_images = inference_images(inference_image_path)
	input_image_size = int(inference_images.get_row_number())*int(inference_images.get_column_number())
	right_count = 0
	batchsize = 1
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(os.path.join(base_path,"train_data/checkPoint/trainModel.meta"))
		saver.restore(sess, tf.train.latest_checkpoint(os.path.join(base_path,"train_data/checkPoint")))

		iterations = inference_images.get_images_number()/batchsize
		for step in range(iterations):
			label_vals = inference_labels.read_labels(batchsize)
			inference_image_pixs = inference_images.read_images(batchsize)
			inference_y_label = []
			for item in label_vals:
				inference_sub_y_label = []
				for i in range(10):
					if item != i:
						inference_sub_y_label.append(0)
					else:
						inference_sub_y_label.append(1)
					inference_y_label.append(inference_sub_y_label)
			inference_x = np.array(inference_image_pixs,dtype=np.float32)
			inference_y = np.array(inference_y_label,dtype=np.float32)
			# 获取需要进行计算的operator
			Ys = sess.graph.get_tensor_by_name('Ys:0')
			X = sess.graph.get_tensor_by_name('X:0')
			Y_ = sess.graph.get_tensor_by_name('Y_:0')
			results = sess.run(Ys,feed_dict={X:inference_x, Y_:inference_y})
			for image_number in range(batchsize):
				maxindex  = np.argmax(results[image_number])
				true_label = np.argmax(inference_y[image_number])
				if maxindex == true_label:
					right_count = right_count + 1

		print("right_count is:{}".format(right_count))
		print("total dataset is:{}".format(inference_images.get_images_number()))
		print("accuracy is:{}".format(float(right_count)/inference_images.get_images_number()))

		# maxindex  = np.argmax(sess.run(op,feed_dict={X:inference_x, Y_:inference_y}))
		# print maxindex  
		# print np.argmax(inference_y[0])