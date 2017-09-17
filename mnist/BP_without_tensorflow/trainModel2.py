import numpy as np
from format_data import preProcess
import math
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x,axis=1,keepdims=True)

def trainModel(epoch,learning_rate):
    filename = 'digitstrain.txt'

    sampleSize= 3000
    batchsize = 3000 # sample size = batchsize * iteration
    iteration = sampleSize/batchsize
    # epoch = 2000

    rawData = np.zeros((sampleSize,785))
    rawImage = np.zeros((batchsize,784))
    rawLabel = np.zeros((batchsize,1))

    #generate the init weight for hidden layer and output layer
    b1 = np.zeros((1,100))
    b2 = np.zeros((1,10))
    w1 = np.random.randn(784,100) / np.sqrt(784)
    w2 = np.random.randn(100,10) / np.sqrt(100)

    # learning_rate = 0.1

    # get the Raw data from the txt file
    with open(filename, 'r+') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            data = lines[i].split(',')
            rawData[i, :] = data  # rawData: (sampleSize,785)
    for k in range(epoch):
        # process random
        np.random.shuffle(rawData)
        for i in range(iteration):
            initValue = i * batchsize
            endValue = (i+1) * batchsize
            # input layer
            rawImage = rawData[initValue:endValue,:-1] # rawImage: (batchsize,784)
            rawLabel = rawData[initValue:endValue, -1] # rawLabel: (batchsize,1)
            rawLabel = rawLabel.astype(np.int64)

            # hidden layer
            neth = rawImage.dot(w1) + b1  #neth: (batchsize,100)
            # outh = sigmoid(neth) #outh: (batchsize,100)
            outh = 1/(1+np.exp(-neth))  #outh: (batchsize,100)

            # output layer
            neto = outh.dot(w2) + b2 #neto: (batchsize,10)
            outo = softmax(neto) #outo: (batchsize,10)

            ## Backpropagation
            ### output layer
            D3 = outo #D3: (batchsize,10)
            ### softmax derivative = D3 -1
            D3[range(batchsize),rawLabel] -= 1
            dw2 = outh.T.dot(D3) / batchsize#dw2: (100,10)
            db2 = np.sum(D3,axis=0,keepdims=True) / batchsize #/batchsize normalization

            D2 = D3.dot(w2.T)*(outh*(1-outh)) #D2 : (3000,100)
            dw1 = rawImage.T.dot(D2) / batchsize# dw1: (784,100)
            db1 = np.sum(D2,axis=0,keepdims=True) / batchsize #/batchsize normalization

            ## update
            w1 += -learning_rate * dw1
            b1 += -learning_rate * db1
            w2 += -learning_rate * dw2
            b2 += -learning_rate * db2

    with open("parameter.txt",'wb+') as fp:
        for i in range(784):
            for j in range(100):
                fp.write("{}\n".format(w1[i,j]))
        for i in range(100):
            for j in range(10):
                fp.write("{}\n".format(w2[i,j]))
        for i in range(100):
            fp.write("{}\n".format(b1[0,i]))
        for i in range(10):
            fp.write("{}\n".format(b2[0,i]))





if __name__ == "__main__":
    trainModel(2200,0.1)