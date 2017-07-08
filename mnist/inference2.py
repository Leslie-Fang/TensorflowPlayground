import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import os
FLAGS = None

def main(inputImageNumber):
    basepath = "./image/figure_"
    inputImage = basepath + inputImageNumber + ".png"
    if os.path.isfile(inputImage):
        pass
    else:
        print("The file doesn't exsit")
        print("exit")
        exit(-1)
    img = cv2.imread(inputImage,0)
    plt.imshow(img,cmap = 'gray', interpolation = 'bicubic')
    plt.show()
    # define input varable
    x = tf.placeholder(tf.float32, [None, 784])
    # define parameter
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    # define the model
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    saver = tf.train.Saver([W, b])
    sess = tf.Session()
    saver.restore(sess, "./model/model.ckpt")
    print("restore the model!")
    print img.shape
    print img
    print("reshape")
    print img.reshape(1,784)
    #inputImg = img.reshape(1,784)
    #print inputImg.shape
    # calculate the result
    ret = sess.run(y, feed_dict={x: img.reshape(1, 784)})
    print(ret)
    print("prediction result:%d"%(ret.argmax()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('-n', action="store", help='an integer for the accumulator', dest='inputImagePath')
    FLAGS, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    main(args.inputImagePath)
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)