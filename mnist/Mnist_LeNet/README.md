## 卷积运算
https://blog.csdn.net/qq_32846595/article/details/79053277
补0的规则
多通道的情况下
https://www.cnblogs.com/lizheng114/p/7498328.html
一张图片通过多个卷积核的计算，可以理解为产生了多个通道

## tf.nn.relu激活函数
https://blog.csdn.net/m0_37870649/article/details/80963053

## tf.nn.max_pool最大值池化函数
https://blog.csdn.net/mzpmzk/article/details/78636184


## how to run
### 准备训练集和测试集
数据集下载地址：http://yann.lecun.com/exdb/mnist/
```
mkdir train_data
cd train_data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gzip -d ##解压两个文件
mkdir test_data
cd test_data
wget ##下载并解压两个训练集合
```

### 运行训练
python mnist_train.py ##在mnist_train.py修改训练的batchsize和epoch
训练结束后模型保存在train_data/checkPoint目录下面

### 运行inference检查训练精度
python mnist_inference.py