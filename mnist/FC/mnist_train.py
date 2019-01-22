# -*- coding: utf-8 -*-
import struct
from PIL import Image
import tensorflow as tf
import numpy as np
import datetime
import os
from train_images import train_images
from train_labels import train_labels
#All the integers in the files are stored in the MSB first (high endian) format used by most non-Intel processors. Users of Intel processors and other low-endian machines must flip the bytes of the header.
#读取二进制文件https://www.jianshu.com/p/3c9137ac5517
#https://www.jianshu.com/p/5a985f29fa81
#二进制文件的格式: http://yann.lecun.com/exdb/mnist/
#数据集
if __name__ == "__main__":
	log_step = 1000
	learning_rate = 0.02
	batchsize = 32
	epoch = 10
	print("Begin train!")
	starttime = datetime.datetime.now()
	base_path = "/home/mnist_dataset/train_data"
	train_image_path = os.path.join(base_path,"train-images-idx3-ubyte")
	train_label_path = os.path.join(base_path,"train-labels-idx1-ubyte")
	train_labels_data = train_labels(train_label_path)
	train_images_data = train_images(train_image_path)
	input_image_size = int(train_images_data.get_row_number())*int(train_images_data.get_column_number())
	X = tf.placeholder(tf.float32,[None,input_image_size],name="X")
	Y_ = tf.placeholder(tf.float32,[None,10],name="Y_") #10表示手写数字识别的10个类别
	W = tf.Variable(tf.zeros([input_image_size,10]),name="weights")
	b = tf.Variable(tf.zeros([10]),name="bias")
	Y = tf.matmul(X,W)+b
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y,labels=Y_,name="loss"))
	Ys = tf.nn.softmax(Y,name="Ys")
	loss = -tf.reduce_sum(Y_ * tf.log(Ys))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)
	count = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for e in range(epoch):
			if train_labels_data is not None:
				del train_labels_data
				train_labels_data = train_labels(train_label_path)
			if train_images_data is not None:
				del train_images_data
				train_images_data = train_images(train_image_path)
			print("train_images_data.get_images_number() is:{}".format(train_images_data.get_images_number()))
			iterations = train_images_data.get_images_number()/batchsize
			for step in range(iterations):
				#label_val = train_labels_data.read_one_label()
				#train_image_pixs = train_images.read_one_image("{0}/{1}_label_{2}.png".format("/home/mnist_dataset/train_data/images",step+1,label_vals[0])) 
				label_vals = train_labels_data.read_labels(batchsize)
				train_image_pixs = train_images_data.read_images(batchsize)
				train_y_label = []
				for item in label_vals:
					train_sub_y_label = []
					for i in range(10):
						if item != i:
							train_sub_y_label.append(0)
						else:
							train_sub_y_label.append(1)
					train_y_label.append(train_sub_y_label)
				train_x = np.array(train_image_pixs,dtype=np.float32)
				train_y = np.array(train_y_label,dtype=np.float32)
				sess.run(train_op,feed_dict={X:train_x, Y_:train_y})
				if (int(step) % int(log_step)) == 0:
					c = sess.run(loss,feed_dict={X:train_x, Y_:train_y})
					print("epoch:{0}, Step:{1}, loss:{2}, W:{3}, b:{4}".format(e+1,step,c,sess.run(W),sess.run(b)))
				# if step >= 50000:
				# 	m = sess.run(Ys,feed_dict={X:train_x, Y_:train_y})
				# 	for item in m:
				# 		maxindex  = np.argmax(item)
				# 		if maxindex == label_val:
				# 			count = count + 1
		print("Running configuration learning_rate:{}, batchsize:{}, epoch:{}".format(learning_rate,batchsize,epoch))
		endtime = datetime.datetime.now()
		print("The program takes:{} sec".format((endtime - starttime).seconds))
		if os.path.isdir(os.path.join(base_path,"checkPoint")) is False:
			os.makedirs(os.path.join(base_path,"checkPoint"))
		saver = tf.train.Saver()
		saver.save(sess,os.path.join(base_path,"checkPoint/trainModel"))
		# print("accuracy is :{}".format(float(count)/(iterations-50000)))





