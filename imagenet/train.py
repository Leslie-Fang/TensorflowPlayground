# -*- coding: utf-8 -*-
import os
from ILSVRC2014_devkit.data import load_meta_clsloc
import cv2
import numpy as np
from prepare_data import readImages
import tensorflow as tf
import datetime

epoch_num = 20
batchsize = 128
log_step = 10
learning_rate = 0.001
start_dir = 0
start_image = 0

#epoch_num:1,time:5600s,accuracy:0

if __name__ == "__main__":
	#build alexnet model
	X = tf.placeholder(tf.float32,[None,227,227,3], name="X_") #shape [None,227,227,3]
	Y_ = tf.placeholder(tf.float32,[None,1000],name="Y_") #10表示手写数字识别的10个类别
	#phase_train = tf.placeholder(tf.bool, name='phase_train')
	#conv1
	layer1_weights = tf.Variable(tf.truncated_normal([11,11,3,96],stddev=0.01))
	layer1_bias = tf.Variable(tf.constant(0.0,shape=[96]))
	layer1_conv = tf.nn.conv2d(X,layer1_weights,strides=[1,4,4,1],padding='VALID')#https://www.cnblogs.com/lizheng114/p/7498328.html
	#计算卷积后图片大小(input size - filter size + 2 * zero padding size) / stride + 1
	#VALID 意味着不padding
	#shape [None,55,55,96]
	layer1_relu = tf.nn.relu(layer1_conv+layer1_bias)#https://blog.csdn.net/m0_37870649/article/details/80963053
	layer1_lrn = tf.nn.local_response_normalization(layer1_relu, 5, 1.0, 0.0001, 0.75)
	layer1_pool = tf.nn.max_pool(layer1_lrn,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')#shape [None,27,27,96]
	#conv1
	layer2_weights = tf.Variable(tf.truncated_normal([5,5,96,256],stddev=0.01))
	layer2_bias = tf.Variable(tf.constant(1.0,shape=[256]))
	layer2_conv = tf.nn.conv2d(layer1_pool,layer2_weights,strides=[1,1,1,1],padding='SAME')#shape [None,27,27,256]
	layer2_relu = tf.nn.relu(layer2_conv+layer2_bias)
	layer2_lrn = tf.nn.local_response_normalization(layer2_relu, 5, 1.0, 0.0001, 0.75)
	layer2_pool = tf.nn.max_pool(layer2_lrn,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')#shape [None,13,13,256]
	#norm
	#<!-----------todo---------->
	# beta = tf.Variable(tf.constant(0.0, shape=[256]), name='beta', trainable=True)
	# gamma = tf.Variable(tf.constant(1.0, shape=[256]),name='gamma', trainable=True)
	# batch_mean, batch_var = tf.nn.moments(layer2_pool, [0,1,2], name='moments')
	# ema = tf.train.ExponentialMovingAverage(decay=0.5)
	# def mean_var_with_update():
	# 	ema_apply_op = ema.apply([batch_mean, batch_var])
	# 	with tf.control_dependencies([ema_apply_op]):
	# 		return tf.identity(batch_mean), tf.identity(batch_var)
	# mean, var = tf.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))
	# normed = tf.nn.batch_normalization(layer2_pool, mean, var, beta, gamma, 1e-3)

	#conv3
	layer3_weights = tf.Variable(tf.truncated_normal([3,3,256,384],stddev=0.01))
	layer3_bias = tf.Variable(tf.constant(0.0,shape=[384]))
	layer3_conv = tf.nn.conv2d(layer2_pool,layer3_weights,strides=[1,1,1,1],padding='SAME')#shape [None,13,13,384]
	layer3_relu = tf.nn.relu(layer3_conv+layer3_bias)

	#conv4
	layer4_weights = tf.Variable(tf.truncated_normal([3,3,384,384],stddev=0.01))
	layer4_bias = tf.Variable(tf.constant(1.0,shape=[384]))
	layer4_conv = tf.nn.conv2d(layer3_relu,layer4_weights,strides=[1,1,1,1],padding='SAME')#shape [None,13,13,384]
	layer4_relu = tf.nn.relu(layer4_conv+layer4_bias)

	#conv5
	layer5_weights = tf.Variable(tf.truncated_normal([3,3,384,256],stddev=0.01))
	layer5_bias = tf.Variable(tf.constant(1.0,shape=[256]))
	layer5_conv = tf.nn.conv2d(layer4_relu,layer5_weights,strides=[1,1,1,1],padding='SAME')#shape [None,13,13,256]
	layer5_relu = tf.nn.relu(layer5_conv+layer5_bias)
	layer5_pool = tf.nn.max_pool(layer5_relu,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')#shape [None,6,6,256]

	#FC1
	layer6_weights = tf.Variable(tf.truncated_normal([6*6*256,4096],stddev=0.01))
	layer6_bias = tf.Variable(tf.constant(0.0,shape=[4096]))
	layer6_flat = tf.reshape(layer5_pool,[-1,6*6*256])#展开成一维，进行全连接层的计算#shape [None,6*6*256]
	layer6_relu = tf.nn.relu(tf.matmul(layer6_flat,layer6_weights)+layer6_bias)#shape [None,4096]

	#dropout1
	keep_prob1 = tf.placeholder(tf.float32,name="keep_prob1")
	h_fc1_drop = tf.nn.dropout(layer6_relu,keep_prob1)

	#FC2:
	layer7_weights = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.01))
	layer7_bias = tf.Variable(tf.constant(0.0,shape=[4096]))
	layer7_relu = tf.nn.relu(tf.matmul(h_fc1_drop,layer7_weights)+layer7_bias)#shape [None,4096]

	#dropout2
	keep_prob2 = tf.placeholder(tf.float32,name="keep_prob2")
	h_fc2_drop = tf.nn.dropout(layer7_relu,keep_prob2)

	#FC3
	layer8_weights = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.01))
	layer8_bias = tf.Variable(tf.constant(0.0,shape=[1000]))
	#Ys = tf.nn.softmax(tf.matmul(h_fc2_drop,layer8_weights)+layer8_bias,name="Ys")#shape [None,1000]
	#layer8_relu = tf.nn.relu(tf.matmul(layer7_relu,layer8_weights)+layer8_bias)#shape [None,1000]
	#y_pred_cls = tf.argmax(Ys,dimension=1)
	layer8_fc = tf.add(tf.matmul(h_fc2_drop,layer8_weights),layer8_bias)
	Ys = tf.nn.softmax(layer8_fc,name="Ys")#shape [None,1000]
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer8_fc, labels=Y_, name='cross-entropy'))
	#L2 regularization（权重衰减）
	#在代价函数后面再加上一个正则化项,http://blog.sina.com.cn/s/blog_a89e19440102x1el.html
	lmbda = 5e-05
	l2_loss = tf.reduce_sum(lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
	loss = cross_entropy + l2_loss
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)	
	# loss = -tf.reduce_mean(Y_*tf.log(Ys))
	# train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	config = tf.ConfigProto()
	config.intra_op_parallelism_threads = 20
	config.inter_op_parallelism_threads = 1
	config.use_per_session_threads = 1
	print('before start')
	starttime = datetime.datetime.now()
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		global_step = 0
		for epoch in range(epoch_num):
			start_dir = 0
			start_image = 0
			step = 0
			while True:
				returen_val = readImages(batchsize,start_dir,start_image)
				if returen_val['statue'] == -1:
					print("returen_val['statue']:-1 Finish epoch:{},step of epoch is:{},global_step is:{},batchsize is:{},learning_rate is {}".format(epoch+1,step,global_step,batchsize,learning_rate))
					break#剩下的图片不够一次运算，直接开始下个epoch
				train_x = np.array(returen_val['data'],dtype=np.float32)
				train_y = np.array(returen_val['label'],dtype=np.float32)
				start_dir = returen_val['return_dir']
				start_image = returen_val['return_image']

				#for test
				# yyy = sess.run(Y_,feed_dict={X:train_x, Y_:train_y, keep_prob1:0.5, keep_prob2:0.5})
				# yyys = sess.run(Ys,feed_dict={X:train_x, Y_:train_y, keep_prob1:0.5, keep_prob2:0.5})
				# print("yyy is:{}".format(yyy))
				# print("yyys is:{}".format(yyys))
				# c = sess.run(loss,feed_dict={X:train_x, Y_:train_y, keep_prob1:0.5, keep_prob2:0.5})
				# print("loss is:{}".format(c))
				# results = sess.run(Ys,feed_dict={X:train_x, Y_:train_y,keep_prob1:0.5, keep_prob2:0.5})
				# for image_number in range(batchsize):
				# 	maxindex  = np.argmax(results[image_number])
				# 	true_label = np.argmax(train_y[image_number])
				# 	print("maxindex is: {}".format(maxindex))
				# 	print("true_label is: {}".format(true_label))
				#exit(1)

				#train
				sess.run(train_op,feed_dict={X:train_x, Y_:train_y, keep_prob1:0.5, keep_prob2:0.5})

				#break

				if (int(step) % int(log_step)) == 0:
					c = sess.run(loss,feed_dict={X:train_x, Y_:train_y, keep_prob1:0.5, keep_prob2:0.5})
					print("epoch:{0}, Step:{1}, loss:{2}".format(epoch+1,step,c))
				step = step + 1
				global_step = global_step + 1
				#if global_step == 
				if returen_val['statue'] == -2:
					print("returen_val['statue']:-2 Finish epoch:{},step of epoch is:{},global_step is:{},batchsize is:{},learning_rate is {}".format(epoch+1,step,global_step,batchsize,learning_rate))
					break#剩下的图片不够下次读取，开始下个epoch
		print("Running configuration learning_rate:{}, batchsize:{}, epoch:{}".format(learning_rate,batchsize,epoch+1))
		endtime = datetime.datetime.now()
		print("The program takes:{} sec".format((endtime - starttime).seconds))
		base_path = os.getcwd()
		if os.path.isdir(os.path.join(base_path,"checkPoint")) is False:
			os.makedirs(os.path.join(base_path,"checkPoint"))
		graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Y_","X_","keep_prob1","keep_prob2","Ys"])
		model_f = tf.gfile.GFile(os.path.join(base_path,"checkPoint/graph.pb"),"wb")
		model_f.write(graph.SerializeToString())
		# Enable tensorboard
		summaryWriter = tf.summary.FileWriter('log/', sess.graph)

