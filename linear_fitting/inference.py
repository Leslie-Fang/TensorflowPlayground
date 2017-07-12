import tensorflow as tf
import argparse

def inference(inputNumber):
    #x should be input data
    x = tf.placeholder(tf.float32)
    W = tf.Variable([1.9], tf.float32)
    b = tf.Variable([0.8], tf.float32)
    #y is calculated by the model
    y = W * x + b
    saver = tf.train.Saver([W, b])
    sess = tf.Session()
    saver.restore(sess, "./model/model.ckpt")
    ret = sess.run(y, feed_dict={x: inputNumber})
    print("The input numer is {}.".format(inputNumber))
    print("The parameter of W and b is {}.".format((sess.run([W, b]))))
    print("The result we inference is {}.".format(ret))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store", help='an integer for the accumulator', dest='inputNumber')
    args = parser.parse_args()
    inference(args.inputNumber)