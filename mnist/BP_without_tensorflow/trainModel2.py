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
    # bn = True is to use batch normalization
    bn = True
    rawData = np.zeros((sampleSize,785))
    rawImage = np.zeros((batchsize,784))
    rawLabel = np.zeros((batchsize,1))

    #generate the init weight for hidden layer and output layer
    np.random.seed(2017)
    b1 = np.zeros((1,100))
    b2 = np.zeros((1,10))
    w1 = np.random.randn(784,100) / np.sqrt(784)
    w2 = np.random.randn(100,10) / np.sqrt(100)
    gamma1 = np.random.uniform(-0.233, 0.233, (3000, 3000))
    beta1 = np.random.uniform(-0.233, 0.233, (3000, 100))
    gamma2 = np.random.uniform(-0.233, 0.233, (3000, 3000))
    beta2 = np.random.uniform(-0.233, 0.233, (3000, 10))
    # learning_rate = 0.1

    # get the Raw data from the txt file
    with open(filename, 'r+') as f:
        lines = f.readlines()
        for i in range(lines.__len__()):
            data = lines[i].split(',')
            rawData[i, :] = data  # rawData: (sampleSize,785)
    for k in range(epoch):
        # process random
        print "epoch : {0}".format(k)
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
            if bn:
                _mean = neth.mean(axis=0, keepdims=True)
                xmuh = (neth - _mean)
                var_2 = ((xmuh) ** 2).mean(axis=0, keepdims=True) + 0.01
                sqrtvarh = np.sqrt(var_2)
                ivarh = 1. / sqrtvarh
                neth = (xmuh) * ivarh
                neth = gamma1.dot(neth)+beta1

            outh = 1/(1+np.exp(-neth))  #outh: (batchsize,100)

            # output layer
            neto = outh.dot(w2) + b2 #neto: (batchsize,10)
            if bn:
                print("use bn")
                _mean = neto.mean(axis=0, keepdims=True)
                xmuo = (neto - _mean)
                var_2 = ((xmuo) ** 2).mean(axis=0, keepdims=True) + 0.01
                sqrtvaro = np.sqrt(var_2)
                ivaro = 1. / sqrtvaro
                neto = (xmuo) * ivaro
                neto = gamma2.dot(neto) + beta2
            outo = softmax(neto) #outo: (batchsize,10)

            ## Backpropagation
            ### output layer

            D3 = outo #D3: (batchsize,10)
            ### softmax derivative = D3 -1
            D3[range(batchsize),rawLabel] -= 1
            if bn:
                # update gamma,beta
                dbeta = np.sum(D3, axis=0)
                dgammax = D3  # not necessary, but more understandable
                print dgammax.shape
                print neto.shape
                print dgammax.dot(neto.T).shape
                dgamma = np.sum(dgammax.dot(neto.T), axis=0)
                print dgamma.shape
                # dneto = dgammax.dot(gamma2)
                dneto = gamma2.dot(dgammax)

                divar = np.sum(dneto * xmuo, axis=0)
                dxmu1 = dneto * ivaro
                dsqrtvar = -1. / (sqrtvaro ** 2) * divar
                dvar = 0.5 * 1. / sqrtvaro * dsqrtvar
                N, D = outo.shape
                dsq = 1. / N * np.ones((N, D)) * dvar

                dxmu2 = 2 * xmuo * dsq

                dx1 = (dxmu1 + dxmu2)
                dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
                dx2 = 1. / N * np.ones((N, D)) * dmu
                dx = dx1 + dx2
                D3 = dx

                print gamma2.shape
                print dgamma.shape
                gamma2 -= learning_rate * dgamma
                beta2 -=  learning_rate * dbeta
                # print("beta2 is:{0}".format(beta2))
                # D3 = gamma2.dot(D3)

            dw2 = outh.T.dot(D3) / batchsize#dw2: (100,10)
            db2 = np.sum(D3,axis=0,keepdims=True) / batchsize #/batchsize normalization

            D2 = D3.dot(w2.T)*(outh*(1-outh)) #D2 : (3000,100)
            if bn:
                # update gamma,beta
                dbeta = np.sum(D2, axis=0)
                dgammax = D2  # not necessary, but more understandable
                dgamma = np.sum(dgammax * neth, axis=0)
                # dneth = dgammax.dot(gamma1)
                dneth = gamma1.dot(dgammax)

                divar = np.sum(dneth * xmuh, axis=0)
                dxmu1 = dneth * ivarh
                dsqrtvar = -1. / (sqrtvarh ** 2) * divar
                dvar = 0.5 * 1. / sqrtvarh * dsqrtvar
                N, D = outh.shape
                dsq = 1. / N * np.ones((N, D)) * dvar

                dxmu2 = 2 * xmuh * dsq

                dx1 = (dxmu1 + dxmu2)
                dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
                dx2 = 1. / N * np.ones((N, D)) * dmu
                dx = dx1 + dx2
                D2 = dx

                gamma1 -= learning_rate * dgamma
                beta1 -= learning_rate * dbeta

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
    trainModel(1000,0.4)