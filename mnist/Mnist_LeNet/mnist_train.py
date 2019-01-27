# -*- coding: utf-8 -*-
import struct
from PIL import Image
import tensorflow as tf
import numpy as np
import datetime
import os
from train_images import train_images
from train_labels import train_labels

if __name__ == "__main__":
	log_step = 1000
	learning_rate = 0.0001
	batchsize = 32
	epoch = 2
	print("Begin train!")
	starttime = datetime.datetime.now()
	base_path = os.path.join(os.getcwd(),"train_data")
	#base_path = "/home/mnist_dataset/train_data"
	train_image_path = os.path.join(base_path,"train-images-idx3-ubyte")
	train_label_path = os.path.join(base_path,"train-labels-idx1-ubyte")
	train_labels_data = train_labels(train_label_path)
	train_images_data = train_images(train_image_path)
	input_image_size = int(train_images_data.get_row_number())*int(train_images_data.get_column_number())
	X = tf.placeholder(tf.float32,[None,input_image_size],name="X")
	x_image = tf.reshape(X,[-1,28,28,1]) #转换成矩阵之后可以进行卷积运算，reshape API https://blog.csdn.net/m0_37592397/article/details/78695318 -1表示由计算过程自动去指定
	Y_ = tf.placeholder(tf.float32,[None,10],name="Y_") #10表示手写数字识别的10个类别
	#conv1
	layer1_weights = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.05))
	layer1_bias = tf.Variable(tf.constant(0.1,shape=[32]))
	layer1_conv = tf.nn.conv2d(x_image,layer1_weights,strides=[1,1,1,1],padding='SAME')#https://www.cnblogs.com/lizheng114/p/7498328.html
	layer1_relu = tf.nn.relu(layer1_conv+layer1_bias)#https://blog.csdn.net/m0_37870649/article/details/80963053
	layer1_pool = tf.nn.max_pool(layer1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #https://blog.csdn.net/mzpmzk/article/details/78636184
	#conv2
	layer2_weights = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.05))
	layer2_bias = tf.Variable(tf.constant(0.1,shape=[64]))
	layer2_conv = tf.nn.conv2d(layer1_pool,layer2_weights,strides=[1,1,1,1],padding='SAME')
	layer2_relu = tf.nn.relu(layer2_conv+layer2_bias)
	layer2_pool = tf.nn.max_pool(layer2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	#FC
	layer3_weights = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.05))
	layer3_bias = tf.Variable(tf.constant(0.1,shape=[1024]))
	layer3_flat = tf.reshape(layer2_pool,[-1,7*7*64])#展开成一维，进行全连接层的计算
	layer3_relu = tf.nn.relu(tf.matmul(layer3_flat,layer3_weights)+layer3_bias)
	#Dropout_layer
	keep_prob = tf.placeholder(tf.float32,name="keep_prob")
	h_fc1_drop = tf.nn.dropout(layer3_relu,keep_prob)
	#FC2
	layer4_weights = tf.Variable(tf.truncated_normal([1024,10],stddev=0.05))
	layer4_bias = tf.Variable(tf.constant(0.1,shape=[10]))
	#Ys = tf.nn.softmax(tf.matmul(layer3_relu,layer4_weights)+layer4_bias,name="Ys")
	Ys = tf.nn.softmax(tf.matmul(h_fc1_drop,layer4_weights)+layer4_bias,name="Ys")  # The output is like [0 0 1 0 0 0 0 0 0 0]
	y_pred_cls = tf.argmax(Ys,dimension=1)
	loss = -tf.reduce_mean(Y_*tf.log(Ys))
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
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
				#sess.run(train_op,feed_dict={X:train_x, Y_:train_y})
				sess.run(train_op,feed_dict={X:train_x, Y_:train_y,keep_prob:0.4})
				if (int(step) % int(log_step)) == 0:
					c = sess.run(loss,feed_dict={X:train_x, Y_:train_y,keep_prob:0.4})
					print("epoch:{0}, Step:{1}, loss:{2}".format(e+1,step,c))
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

	