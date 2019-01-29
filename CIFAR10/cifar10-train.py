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

IMAGE_DEPTH = 32
IMAGE_WIDTH = 32
IMAGE_CHANNEL = 3
IMAGE_SIZE = IMAGE_DEPTH * IMAGE_WIDTH * IMAGE_CHANNEL

epoch_num = 20
batchsize = 32
log_step = 100
learning_rate = 0.0001

# epochnum:2,accuray:0.53
# epochnum:5,accuray:0.6021

def unpickle(file):
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
	meta_data = unpickle(os.path.join(args.dataset,"batches.meta"))
	label_names = meta_data['label_names']
	print(label_names)
	## 构造神经网络
	X = tf.placeholder(tf.float32,[None,IMAGE_SIZE], name="X_")
	x_image = tf.reshape(X,[-1,32,32,3]) #转换成矩阵之后可以进行卷积运算，reshape API https://blog.csdn.net/m0_37592397/article/details/78695318 -1表示由计算过程自动去指定
	Y_ = tf.placeholder(tf.float32,[None,10],name="Y_") #10表示手写数字识别的10个类别
	#conv1
	layer1_weights = tf.Variable(tf.truncated_normal([5,5,3,64],stddev=0.05))
	layer1_bias = tf.Variable(tf.constant(0.1,shape=[64]))
	layer1_conv = tf.nn.conv2d(x_image,layer1_weights,strides=[1,1,1,1],padding='SAME')#https://www.cnblogs.com/lizheng114/p/7498328.html
	layer1_relu = tf.nn.relu(layer1_conv+layer1_bias)#https://blog.csdn.net/m0_37870649/article/details/80963053
	layer1_pool = tf.nn.max_pool(layer1_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') #https://blog.csdn.net/mzpmzk/article/details/78636184
	#conv2
	layer2_weights = tf.Variable(tf.truncated_normal([5,5,64,128],stddev=0.05))
	layer2_bias = tf.Variable(tf.constant(0.1,shape=[128]))
	layer2_conv = tf.nn.conv2d(layer1_pool,layer2_weights,strides=[1,1,1,1],padding='SAME')
	layer2_relu = tf.nn.relu(layer2_conv+layer2_bias)
	layer2_pool = tf.nn.max_pool(layer2_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	#FC1
	layer3_weights = tf.Variable(tf.truncated_normal([8*8*128,1024],stddev=0.05))
	layer3_bias = tf.Variable(tf.constant(0.1,shape=[1024]))
	layer3_flat = tf.reshape(layer2_pool,[-1,8*8*128])#展开成一维，进行全连接层的计算
	layer3_relu = tf.nn.relu(tf.matmul(layer3_flat,layer3_weights)+layer3_bias)
	#FC2
	layer4_weights = tf.Variable(tf.truncated_normal([1024,512],stddev=0.05))
	layer4_bias = tf.Variable(tf.constant(0.1,shape=[512]))
	layer4_relu = tf.nn.relu(tf.matmul(layer3_relu,layer4_weights)+layer4_bias)
	#FC3
	layer5_weights = tf.Variable(tf.truncated_normal([512,10],stddev=0.05))
	layer5_bias = tf.Variable(tf.constant(0.1,shape=[10]))
	Ys = tf.nn.softmax(tf.matmul(layer4_relu,layer5_weights)+layer5_bias,name="Ys")  # The output is like [0 0 1 0 0 0 0 0 0 0]
	y_pred_cls = tf.argmax(Ys,dimension=1)
	loss = -tf.reduce_mean(Y_*tf.log(Ys))
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	#完成神经网络构造
	starttime = datetime.datetime.now()
	start_point = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(epoch_num):
			step = 0
			for Train_file in Train_files:
				train_data_dict = unpickle(Train_file)
				while True:
					input_data = read_images(train_data_dict['data'],train_data_dict['labels'],batchsize,start_point)
					if input_data == -1:
						start_point = 0
						break
					start_point = start_point + batchsize
					train_x = np.array(input_data["data"],dtype=np.float32)
					train_y = np.array(input_data["label"],dtype=np.float32)
					# print train_x.shape
					# print train_y.shape
					# print train_y[0]
					sess.run(train_op,feed_dict={X:train_x, Y_:train_y})
					if (int(step) % int(log_step)) == 0:
						c = sess.run(loss,feed_dict={X:train_x, Y_:train_y})
						print("epoch:{0}, Step:{1}, loss:{2}".format(epoch+1,step,c))
					step = step + 1
		print("Running configuration learning_rate:{}, batchsize:{}, epoch:{}".format(learning_rate,batchsize,epoch))
		endtime = datetime.datetime.now()
		print("The program takes:{} sec".format((endtime - starttime).seconds))
		base_path = os.getcwd()
		if os.path.isdir(os.path.join(base_path,"checkPoint")) is False:
			os.makedirs(os.path.join(base_path,"checkPoint"))
		# saver = tf.train.Saver()
		# saver.save(sess,os.path.join(base_path,"checkPoint/trainModel"))
		# print("accuracy is :{}".format(float(count)/(iterations-50000)))
		graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Y_","X_","Ys"])
		model_f = tf.gfile.GFile(os.path.join(base_path,"checkPoint/graph.pb"),"wb")
		model_f.write(graph.SerializeToString())
		# Enable tensorboard
		summaryWriter = tf.summary.FileWriter('log/', sess.graph)
