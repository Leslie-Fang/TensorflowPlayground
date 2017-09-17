import numpy as np
from format_data import preProcess,preProcess2
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def verifyModel(image,w1, w2,b1,b2):
    # input layer
    rawData = np.array(image).reshape((784,), order='C')

    # hidden layer
    neth = np.dot(rawData.T, w1) + b1
    outh = sigmoid(neth)

    # output layer
    neto = np.dot(outh, w2) + b2
    outo = softmax(neto)
    # y_ if the prediction digit
    y_ = outo.argmax()
    return y_

def verifyData():
    filename = 'digitstest.txt'
    w1 = np.random.random((784,100))
    w2 = np.random.random((100, 10))
    b1 = np.random.random((100))
    b2 = np.random.random((10))
    with open("parameter.txt",'r+') as fp:
        for i in range(784):
            for j in range(100):
                w1[i,j] = float(fp.readline().strip())
        for i in range(100):
            for j in range(10):
                w2[i,j] = float(fp.readline().strip())
        for i in range(100):
            b1[i] = float(fp.readline().strip())
        for i in range(10):
            b2[i] = float(fp.readline().strip())

    with open(filename,'r+') as f:
        lines = f.readlines()
        totalCase = 0.0
        correct = 0.0
        # image, label = preProcess2(lines[1000])
        # y_predict = verifyModel(image, w1, w2, b1, b2)
        # print("The number is: {}".format(y_predict))
        for i in range(lines.__len__()):
            image, label = preProcess(lines[i])
            y_predict = verifyModel(image, w1, w2,b1,b2)
            print("The label is: {}".format(label))
            print("The prediction is: {}".format(y_predict))
            totalCase = totalCase + 1.0
            print label
            print y_predict
            if abs(label - y_predict) < 0.1:
                correct = correct + 1.0
                print("=======>")
        print("Total test number is: {}".format(totalCase))
        print("Success test number is: {}".format(correct))
        print("Success Rate is: {}".format(correct/totalCase))
    return correct/totalCase

if __name__ == "__main__":
    verifyData()




