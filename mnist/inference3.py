import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import cv2
import os
FLAGS = None

# comparing with inference2, inference3 is used to call in the gui.py
def restoreModel():
    # define parameter
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    saver = tf.train.Saver([W, b])
    sess = tf.Session()
    saver.restore(sess, "./model/model.ckpt")
    return sess,W,b

def main(inputImagepath,inputImagename,sess,W,b):
    basepath = "./image/"
    inputImage = basepath + inputImagename
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
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    #print("restore the model!")
    #print img.shape
    print "The img is: "
    print img
    #print("reshape")
    #print img.reshape(1,784)
    ret = sess.run(y, feed_dict={x: img.reshape(1, 784)})
    print(ret)
    print("prediction result:%d"%(ret.argmax()))
    return ret.argmax()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    parser.add_argument('-n', action="store", help='an integer for the accumulator', dest='inputImagePath')
    FLAGS, unparsed = parser.parse_known_args()
    args = parser.parse_args()
    main(args.inputImagePath)
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)