## 准备训练集和测试集
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

## 运行训练
python mnist_train.py ##在mnist_train.py修改训练的batchsize和epoch
训练结束后模型保存在train_data/checkPoint目录下面

## 运行inference检查训练精度
python mnist_inference.py

训练参数
* learning_rate = 0.02
* batchsize = 32
* epoch = 10
的情况下，inference的精度可以到90%以上