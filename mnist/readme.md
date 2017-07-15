这个仓库的代码主要和使用tensorflow进行手写数字识别相关
相关代码主要包含了一下几个部分的内容

## 图像预处理
preprocessimage.py
读取摄像头拍摄到的手写数字的图片
对图片进行[预处理](!https://leslie-fang.github.io/2017/07/06/手写数字识别/)
保存预处理之后的图片到 /image目录下面

## 使用softmax进行手写数字识别
模型保存在 model的目录下面
* inference2.py
输入为image目录下面的预处理之后的图片，调用训练之后的模型，得到训练结果
使用方法
```
python inference2.py -n 1
```
* gui
将上述函数和图像的拍摄、保存、预处理以及识别放在一个gui中
相关的代码：
gui.py
inference3.py
preprocessimage.py
使用方法
```
python gui.py
```

## 使用CNN进行手写数字识别
模型保存在 model2的目录下面,整个模型的[介绍](!https://leslie-fang.github.io/2017/07/15/卷积神经网络/)
* inference_deep.py
输入为image目录下面的预处理之后的图片，调用训练之后的模型，得到训练结果
使用方法
```
python inference_deep.py -n 1
```
* gui
将上述函数和图像的拍摄、保存、预处理以及识别放在一个gui中
相关的代码：
gui_deep.py
inference_deep2.py
preprocessimage.py
使用方法
```
python gui_deep.py
```

## python调用C++的动态链接库
相关代码
makefile
add.cpp
testPythonCpp.py
使用方法：
```
make #编译add.cpp
python testPythonCpp.py
```
