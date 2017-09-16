import numpy as np
import matplotlib.pyplot as plt

def myFileOpen(filename):
    with open(filename,'r+') as f:
        lines = f.readlines()
        print(lines.__len__())
        # for line in lines:
        #     print("==========>")
        #     print(line)
        #     print(line.split(',').__len__())

def processSingleImage(filename):
    with open(filename,'r+') as f:
        lines = f.readlines()
        line = lines[1895]
        print(line)
        print(line.split(',').__len__())
        image = line.split(',')[:-1]
        label = line.split(',')[-1]
        for i in range(image.__len__()):
            print i
            image[i] = float(image[i])
        label = int(label)
        print label
        print image.__len__()
        rawData = np.array(image).reshape((28,28),order='C')
        print rawData
        print rawData.dtype
        print("rawData.size is : {}".format(rawData.shape))
        print("The label is: {}".format(label))
        plt.imshow(rawData, cmap='gray')
        plt.show()

def preProcess(line):
    image = line.split(',')[:-1]
    label = line.split(',')[-1]
    for i in range(image.__len__()):
        image[i] = float(image[i])
    label = int(label)
    # rawData = np.array(image).reshape((28, 28), order='C')
    # plt.imshow(rawData, cmap='gray')
    # plt.show()
    return image,label

if __name__ == "__main__":
    myFileOpen('digitstrain.txt')
    processSingleImage('digitstrain.txt')