import numpy as np
from format_data import preProcess
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def trainModel(image,label,learning_rate,w1, w2):
    # input layer
    rawData = np.array(image).reshape((784,), order='C')

    # hidden layer
    neth = np.dot(rawData.T, w1) + b1
    outh = sigmoid(neth)

    # output layer
    neto = np.dot(outh.T, w2) + b2
    outo = softmax(neto)
    # y_ if the prediction digit
    y_ = outo.argmax()

    ## Backpropagation
    ### output layer
    target = np.zeros((10,))
    target[label] = 1
    # print target
    # print outo
    Etotal = 0
    for i in range(outo.shape[0]):
        Etotal = Etotal + math.pow((outo[i] - target[i]), 2)
    # print Etotal
    dw2 = np.zeros((100, 10))
    for i in range(dw2.shape[0]):
        for j in range(dw2.shape[1]):
            dw2[i, j] = -(target[j] - outo[j]) * outo[j] * (1 - outo[j]) * outh[j]
            w2[i, j] = w2[i, j] - learning_rate * dw2[i, j]

    # print w2

    ### hidden layer
    dw1 = np.zeros((784, 100))
    for i in range(dw1.shape[0]):
        for j in range(dw1.shape[1]):
            suberror = 0
            for z in range(10):
                suberror = suberror + (-(target[z] - outo[z]) * outo[z] * (1 - outo[z]) * w2[j, z])
            dw1[i, j] = suberror * outh[j] * (1 - outh[j]) * rawData[i]
            w1[i, j] = w1[i, j] - learning_rate * dw1[i, j]
    # print w1
    return w1,w2,y_,Etotal


if __name__ == "__main__":
    filename = 'digitstrain.txt'

    #generate the init weight for hidden layer and output layer
    b1 = np.random.random((100))
    b2 = np.random.random((10))
    w1 = np.random.random((784,100))
    w2 = np.random.random((100, 10))
    learning_rate = 0.1
    # print("b1 is: {}".format(b1))
    # print("b2 is: {}".format(b2))
    # print("w1 is: {}".format(w1))
    # print("w1 is: {}".format(w2))
    # print b1.shape
    # print b2.shape
    # print w1.shape
    # print (np.dot(w1,w1)).shape
    with open(filename,'r+') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            image, label = preProcess(lines[i])
            w1,w2,y_,Etotal = trainModel(image, label, learning_rate, w1, w2)
            print("The label is: {}".format(label))
            print("The prediction is: {}".format(y_))
            print("The totalError is: {}".format(Etotal))
        # image,label = preProcess(lines[50])
        # trainModel(image,label,learning_rate,w1,w2)
