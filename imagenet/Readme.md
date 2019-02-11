Imagenet 官网：
http://www.image-net.org/about-overview

中文数据库解释:
https://cloud.tencent.com/developer/article/1010187

## 准备数据
* 参考官网：http://www.image-net.org/download-imageurls
* wget http://www.image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
* tar zxvf 得到一个txt文档，里面是所有图片的下载地址"fall11_urls.txt"
* The URLs are listed in a single txt file, where each line contains an image ID and the original URL. The image ID is formatted as xxxx_yyyy, where xxxx represents the WordNet ID (wnid) of this image. If you download the original image files, the image ID is the same as the filename ( excluding the format extension ).
* 看一行数据:n01397114_3172  http://farm4.static.flickr.com/3437/3189448810_e8473682f1.jpg
* 下载地址就是http://farm4.static.flickr.com/3437/3189448810_e8473682f1.jpg， wordnet Id:n01397114
* 通过URL去查看图片:http://www.image-net.org/synset?wnid=(id)
* 列子:http://www.image-net.org/synset?wnid=n01397114

数据集的中文介绍:
https://www.zhihu.com/question/273633408/answer/369134332
标签值和含义之间的对应关系:
https://gist.github.com/maraoz/388eddec39d60c6d52d4
http://www.image-net.org/download-API

## 或者直接下载imagenet 2014的数据集合(**推荐使用这种方式**)
http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php
数据格式
There are a total of 456567 images for training. The number of positive images for each synset (category) ranges from 461 to 67513. The number of negative images ranges from 42945 to 70626 per synset. There are 20121 validation images, and 40152 test images. All images are in JPEG format.

1. 下载train，validation和test数据集合
2. 数据的含义下载链接里面的develop-kit这个链接
解压之后里面有个readme.txt，去解析所有的文件内容


## 正负样本
绿色箭头指向positive pair（一张是source domain的原图，一张是生成的图片），红色箭头指向negative pair（一张是生成的图片，一张是target domain的图片）。这里的学习目标也是要让正对的距离小（即，self-similarity ），负对的距离大（即，domain-dissimilarity）。
https://www.zhihu.com/question/67616216

## 目标检测算法
参考：https://www.zhihu.com/question/53438706
https://zhuanlan.zhihu.com/ML-Algorithm
代码参考: https://github.com/tensorflow/models/tree/master/research/object_detection
深度方法主要分为single-stage(e.g. SSD, YOLO)和two-stage(e.g. RCNN系列)两种. single-stage直接在图片上经过计算生成detections. two-stage先提proposal, 再基于proposal做二次修正. 相对来说single-stage速度快, 精度低. 而two-stage精度高, 速度慢.

神经网络：
1. RetinaNet
2. Faster-RCNN(with FPN)
https://zhuanlan.zhihu.com/p/31426458

## 分类模型
参考论文：
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
中文翻译：
https://blog.csdn.net/LK274857347/article/details/53514364

这里使用AlexNet,是2012年比赛的冠军
Alexnet模型的组成：
https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637

参考代码：
https://github.com/ryujaehun/alexnet/blob/master/codes/train.py

BN层(Batch Normalization)：
介绍：https://zhuanlan.zhihu.com/p/29957294
TF API：

## 读取分类图片
使用python opencv
