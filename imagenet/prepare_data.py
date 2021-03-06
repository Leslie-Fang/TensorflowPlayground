# -*- coding: utf-8 -*-
#参考官网：http://www.image-net.org/download-imageurls
#wget http://www.image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
#tar zxvf 得到一个txt文档，里面是所有图片的下载地址"fall11_urls.txt"
#The URLs are listed in a single txt file, where each line contains an image ID and the original URL. The image ID is formatted as xxxx_yyyy, where xxxx represents the WordNet ID (wnid) of this image. If you download the original image files, the image ID is the same as the filename ( excluding the format extension ).
# 看一行数据:n01397114_3172  http://farm4.static.flickr.com/3437/3189448810_e8473682f1.jpg
# 下载地址就是http://farm4.static.flickr.com/3437/3189448810_e8473682f1.jpg， wordnet Id:n01397114
# 通过URL去查看图片:http://www.image-net.org/synset?wnid=(id)
# 列子:http://www.image-net.org/synset?wnid=n01397114
import os
from ILSVRC2014_devkit.data import load_meta_clsloc
import cv2
import numpy as np
import threading
import Queue

TrainData_base_Url = "/home/lesliefang/il2014_CLS_LOC/train_data"
dirs = os.listdir(TrainData_base_Url)
meta_file = os.path.join(os.getcwd(),"ILSVRC2014_devkit/data/meta_clsloc.mat")
WNID2ID = load_meta_clsloc.getData(meta_file)
que = Queue.Queue()

def untarAllTrainData():
	#运行一次，用于解压所有train的数据
	for root, dirs, files in os.walk(TrainData_base_Url):
		# print("root", root)  # 当前目录路径
		# print("dirs", dirs)  # 当前路径下所有子目录
		# print("files", files)  # 当前路径下所有非目录子文件
		pass
	for file in files:
		print file
		directory = file.split('.')[0]
		print directory
		os.system("cd {0} && mkdir {1} && tar xvf {2} -C {3} && rm -rf {4}".format(TrainData_base_Url,directory,file,directory,file))

def sub_readImages_v2(batchsize, start_dir, start_image):
	sub_return_images = []
	sub_return_labels = []
	sub_returen_val = {}
	readed_image_number = 0
	global que
	for i in range(start_dir,len(dirs)):
		dir = dirs[i]
		label_value = WNID2ID[dir]
		label = np.zeros((1000,), dtype=int) #总共1000个类别
		label[label_value-1] = 1
		dir_path = os.path.join(TrainData_base_Url,dir)
		images = os.listdir(dir_path)
		for j in range(start_image,len(images)):
			image = images[j]
			img = cv2.imread(os.path.join(dir_path,image))
			img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
			single_img1 = (img.astype(np.float32)/255.0).tolist()#优化图片读取时的转换过程

			sub_return_images.append(single_img1)
			sub_return_labels.append(label)
			readed_image_number = readed_image_number + 1
			if readed_image_number >= batchsize:
				sub_returen_val['statue'] = 1
				if (j+1) < len(images):
					return_dir = i
					return_image = j + 1
				else:
					return_dir = i + 1
					if return_dir >= len(dirs):
						sub_returen_val['statue'] = -2#正好读完所有图片，不支持下一次去读了
					return_image = 0
				sub_returen_val['data'] = sub_return_images
				sub_returen_val['label'] = sub_return_labels
				sub_returen_val['return_dir'] = return_dir
				sub_returen_val['return_image'] = return_image
				que.put(sub_returen_val)
				return sub_returen_val
			if j >= len(images)-1:
				#移动到下一个dir的时候从头开始读取图片
				start_image = 0
	if readed_image_number < batchsize:
		sub_returen_val['statue'] = -1 #剩下的所有图片都不够1个batchsize的情况下
		que.put(sub_returen_val)
		return sub_returen_val

def readImages_v2(batchsize, start_dir, start_image):
	returen_val = {}
	returen_val['data'] = []
	returen_val['label'] = []
	thread_num = 4
	threads = []
	sub_batchsize = batchsize/thread_num

	start_dirs = []
	start_images = []
	#获得每个线程应该读取的图片的起点位置
	start_dirs.append(start_dir)
	start_images.append(start_image)

	temp_start_dir = start_dirs[0]
	temp_start_image = start_images[0]
	temp_batchsize = sub_batchsize
	while True:
		dir = dirs[temp_start_dir]
		dir_path = os.path.join(TrainData_base_Url,dir)
		images = os.listdir(dir_path)
		left_image_num = len(images)-temp_start_image
		if left_image_num > temp_batchsize:
			start_dirs.append(temp_start_dir)
			start_images.append(temp_start_image + temp_batchsize)
			break
		else:
			temp_start_dir = temp_start_dir + 1
			if temp_start_dir >= len(dirs):
				returen_val['statue'] = -1 #剩下的所有图片都不够1个batchsize的情况下
				return returen_val
			temp_start_image = 0
			temp_batchsize = temp_batchsize - left_image_num

	temp_start_dir = start_dirs[1]
	temp_start_image = start_images[1]
	temp_batchsize = sub_batchsize
	while True:
		dir = dirs[temp_start_dir]
		dir_path = os.path.join(TrainData_base_Url,dir)
		images = os.listdir(dir_path)
		left_image_num = len(images)-temp_start_image
		if left_image_num > temp_batchsize:
			start_dirs.append(temp_start_dir)
			start_images.append(temp_start_image + temp_batchsize)
			break
		else:
			temp_start_dir = temp_start_dir + 1
			if temp_start_dir >= len(dirs):
				returen_val['statue'] = -1 #剩下的所有图片都不够1个batchsize的情况下
				return returen_val
			temp_start_image = 0
			temp_batchsize = temp_batchsize - left_image_num

	temp_start_dir = start_dirs[2]
	temp_start_image = start_images[2]
	temp_batchsize = sub_batchsize
	while True:
		dir = dirs[temp_start_dir]
		dir_path = os.path.join(TrainData_base_Url,dir)
		images = os.listdir(dir_path)
		left_image_num = len(images)-temp_start_image
		if left_image_num > temp_batchsize:
			start_dirs.append(temp_start_dir)
			start_images.append(temp_start_image + temp_batchsize)
			break
		else:
			temp_start_dir = temp_start_dir + 1
			if temp_start_dir >= len(dirs):
				returen_val['statue'] = -1 #剩下的所有图片都不够1个batchsize的情况下
				return returen_val
			temp_start_image = 0
			temp_batchsize = temp_batchsize - left_image_num

	return_images = []
	return_labels = []
	global que
	with que.mutex:
		que.queue.clear()
	for i in range(thread_num):
		t = threading.Thread(target=sub_readImages_v2,args=(sub_batchsize,start_dirs[i],start_images[i]))
		threads.append(t)
	for i in range(thread_num):
		threads[i].start()
	for i in range(thread_num):
		threads[i].join()

	for i in range(thread_num):
		result = que.get()
		if result['statue'] != 1:
			returen_val['statue'] = -1
			break
		else:
			returen_val['statue'] = 1
		if i == 3:
			returen_val['return_dir'] = result['return_dir']
			returen_val['return_image'] = result['return_image']
		for item in result['data']:
			returen_val['data'].append(item)
		for item in result['label']:
			returen_val['label'].append(item)

	return returen_val

def readImages(batchsize, start_dir, start_image):
	return_images = []
	return_labels = []
	returen_val = {}
	readed_image_number = 0
	for i in range(start_dir,len(dirs)):
		dir = dirs[i]
		label_value = WNID2ID[dir]
		# print(dir)
		# print(label_value)
		label = np.zeros((1000,), dtype=int) #总共1000个类别
		label[label_value-1] = 1
		# print(label[999])
		# print(label[label_value-1])
		dir_path = os.path.join(TrainData_base_Url,dir)
		images = os.listdir(dir_path)
		for j in range(start_image,len(images)):
			image = images[j]
			# print(image)
			img = cv2.imread(os.path.join(dir_path,image))
			# print(img.shape)#每张图片的shape都不一样，统一缩放到227*227*3
			img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)#每张图片的shape都不一样，统一缩放到227*227*3
			# single_img1 = []
			# single_img2 = []
			# single_img3 = []
			# for m in range(len(img)):
			# 	single_img2 = []
			# 	for p in range(len(img[0])):
			# 		single_img3 = []
			# 		for q in range(len(img[0][0])):
			# 			niii = float(img[m][p][q])/255.0#很重要，一定要归一到255
			# 			single_img3.append(niii)
			# 		single_img2.append(single_img3)
			# 	single_img1.append(single_img2)
			single_img1 = (img.astype(np.float32)/255.0).tolist()#优化图片读取时的转换过程

			return_images.append(single_img1)
			return_labels.append(label)
			readed_image_number = readed_image_number + 1
			if readed_image_number >= batchsize:
				returen_val['statue'] = 1
				if (j+1) < len(images):
					return_dir = i
					return_image = j + 1
				else:
					return_dir = i + 1
					if return_dir >= len(dirs):
						returen_val['statue'] = -2#正好读完所有图片，不支持下一次去读了
					return_image = 0
				returen_val['data'] = return_images
				returen_val['label'] = return_labels
				returen_val['return_dir'] = return_dir
				returen_val['return_image'] = return_image
				return returen_val
			if j >= len(images)-1:
				#移动到下一个dir的时候从头开始读取图片
				start_image = 0 
	if readed_image_number < batchsize:
		returen_val['statue'] = -1 #剩下的所有图片都不够1个batchsize的情况下
		return returen_val

if __name__ == "__main__":
	#untarAllTrainData()
	batchsize = 2048
	start_dir = 0
	start_image = 0
	for i in range(2):
		returen_val = readImages(batchsize,start_dir,start_image)
		# print(returen_val['data'].shape)
		train_x = np.array(returen_val['data'],dtype=np.float32)
		train_y = np.array(returen_val['label'],dtype=np.float32)
		start_dir = returen_val['return_dir']
		start_image = returen_val['return_image']
		print(train_x.shape)#(batchsize, 227, 227, 3)
		print(train_y.shape)#(batchsize, 1000)
		print(start_dir)
		print(start_image)




