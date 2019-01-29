## kaggle
很好的数据科学相关的网站
kaggle上有关于cifar10的挑战赛
https://www.kaggle.com/c/cifar-10

## 下载数据集
https://www.cs.toronto.edu/~kriz/cifar.html
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

## How to run train
下载并解压数据集到/root/testcode/dataset/cifar10/cifar-10-batches-py 这个目录下面
python CIFAR10/cifar10-train.py -d /root/testcode/dataset/cifar10/cifar-10-batches-py

## How to run inference
python CIFAR10/cifar10-inference.py -d /root/testcode/dataset/cifar10/cifar-10-batches-py

## int8化
训练生成的pb文件保存在/root/testcode/tf_testcode/checkPoint/graph.pb
运行int8的计算，用private-tensorflow的int8 master的分支
python $TF_ROOT/tensorflow/tools/quantization/quantize_graph.py --input=/root/testcode/tf_testcode/checkPoint/graph.pb --output=int8_graph.pb --output_node_names='X_,Y_,Ys' --print_nodes --mode=eightbit --intel_cpu_eightbitize=True

### 和FP32一样的方法跑inference
python CIFAR10/cifar10-inference.py -d /root/testcode/dataset/cifar10/cifar-10-batches-py
精度：0.6021
内存的使用
```
[root@localhost private-tensorflow]# free -m
              total        used        free      shared  buff/cache   available
Mem:         385660        1418      382042          19        2199      381724
Swap:          4095           0        4095

```

### 和INT8一样的方法跑inference
python CIFAR10/cifar10-inference.py -d /root/testcode/dataset/cifar10/cifar-10-batches-py -g checkPoint/int8_graph.pb
精度：0,6025
内存的使用
```
[root@localhost private-tensorflow]# free -m
              total        used        free      shared  buff/cache   available
Mem:         385660        1386      382074          19        2198      381756
Swap:          4095           0        4095
```

正常不跑测试时：
```
[root@localhost private-tensorflow]# free -m
              total        used        free      shared  buff/cache   available
Mem:         385660        1064      382398          19        2197      382079
Swap:          4095           0        4095
```

## Next Step
1. 网络结构可以改进
2. 读取图片的过程可以优化
