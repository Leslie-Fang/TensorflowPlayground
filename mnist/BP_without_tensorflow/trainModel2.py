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
                var_2 = ((xmuh) ** 2).mean(axis=0) + 0.01
                sqrtvarh = np.sqrt(var_2)
                ivarh = 1. / sqrtvarh
                nethm = (xmuh) * ivarh
                neth = gamma1.dot(nethm)+beta1

            outh = 1/(1+np.exp(-neth))  #outh: (batchsize,100)

            # output layer
            neto = outh.dot(w2) + b2 #neto: (batchsize,10)
            if bn:
                print("use bn")
                _mean = neto.mean(axis=0, keepdims=True)
                xmuo = (neto - _mean) #xmuo (3000,10)
                # print("xmuo : {0}".format(xmuo.shape))
                var_2 = ((xmuo) ** 2).mean(axis=0) + 0.01 #var_2 (10)
                # print("var_2 : {0}".format(var_2.shape))
                sqrtvaro = np.sqrt(var_2)
                # print("sqrtvaro : {0}".format(sqrtvaro.shape))
                ivaro = 1. / sqrtvaro
                # print("ivaro : {0}".format(ivaro.shape))
                netom = (xmuo) * ivaro
                neto = gamma2.dot(netom) + beta2
            outo = softmax(neto) #outo: (batchsize,10)

            ## Backpropagation
            ### output layer

            D3 = outo #D3: (batchsize,10)
            ### softmax derivative = D3 -1
            D3[range(batchsize),rawLabel] -= 1
            if bn:
                # update gamma,beta
                N, D = D3.shape
                #step 9
                dbeta = np.sum(D3, axis=0) #dbeta (10,)
                dgammax = D3  # not necessary, but more understandable #dgammax: (3000,10)

                #step 8
                dgamma = np.sum(dgammax.dot(neto.T), axis=0) #dgamma: (3000,)
                # dneto = dgammax.dot(gamma2)
                # print("dgammax : {0}".format(dgammax.shape))
                # print("gamma2 : {0}".format(gamma2.shape))
                dneto = gamma2.dot(dgammax) #dneto: (3000:10)

                #step 7
                # print("dneto : {0}".format(dneto.shape))
                # print("xmuo : {0}".format(xmuo.shape))
                divar = np.sum(dneto.T.dot(xmuo), axis=0) #divar: (10,)
                # print("divar : {0}".format(divar.shape))
                # print("dneto : {0}".format(dneto.shape))
                # print("ivaro : {0}".format(ivaro.shape))
                dxmu1 = dneto * ivaro  #dxmu1: (3000,10)
                # print("dxmu1 : {0}".format(dxmu1.shape))

                #step 6
                dsqrtvar = -1. / (sqrtvaro ** 2) * divar #dsqrtvar : (10,)
                # print("dsqrtvar : {0}".format(dsqrtvar.shape))

                # step5
                dvar = 0.5 * 1. / sqrtvaro * dsqrtvar  #dvar : (10,)
                # print("dvar : {0}".format(dvar.shape))

                # step4
                dsq = 1. / N * np.ones((N, D)) * dvar #dsq : (3000, 10)
                # print("dsq : {0}".format(dsq.shape))

                # step3
                dxmu2 = 2 * xmuo * dsq  #dxmu2 : (3000, 10)
                # print("dxmu2 : {0}".format(dxmu2.shape))

                # step2
                dx1 = (dxmu1 + dxmu2)  #dx1 : (3000, 10)
                dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)  # dmu : (10,)
                # print("dx1 : {0}".format(dx1.shape))
                # print("dmu : {0}".format(dmu.shape))

                # step1
                dx2 = 1. / N * np.ones((N, D)) * dmu #dx2 : (3000, 10)
                # print("dx2 : {0}".format(dx2.shape))

                # step0
                dx = dx1 + dx2  #dx : (3000, 10)
                # print("dx : {0}".format(dx.shape))
                D3 = dx

                # print("gamma2 : {0}".format(gamma2.shape))
                # print("dgamma : {0}".format(dgamma.shape))
                gamma2 -= learning_rate * dgamma
                beta2 -=  learning_rate * dbeta


            dw2 = outh.T.dot(D3) / batchsize#dw2: (100,10)
            db2 = np.sum(D3,axis=0,keepdims=True) / batchsize #/batchsize normalization

            D2 = D3.dot(w2.T)*(outh*(1-outh)) #D2 : (3000,100)
            if bn:
                # update gamma,beta
                N, D = D2.shape
                dbeta = np.sum(D2, axis=0)
                dgammax = D2  # not necessary, but more understandable
                dgamma = np.sum(dgammax.dot(neth.T), axis=0)
                # dneth = dgammax.dot(gamma1)
                dneth = gamma1.dot(dgammax)

                divar = np.sum(dneth.T.dot(xmuh), axis=0)
                dxmu1 = dneth * ivarh
                dsqrtvar = -1. / (sqrtvarh ** 2) * divar
                dvar = 0.5 * 1. / sqrtvarh * dsqrtvar

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
    trainModel(100,0.4)