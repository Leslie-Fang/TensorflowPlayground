import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
import cv2

FLAGS = None

def main(_):
    # Import data
    #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    img = cv2.imread("res.png",0);
    plt.imshow(img)
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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)