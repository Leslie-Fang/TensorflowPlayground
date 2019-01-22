import numpy as np
import tensorflow as tf

# the model should be y = W*x + b, moreover, we set the model as y = 2*x + 1
# x is the input the data, y should be the output
# y should be calculated, but we don't know the true value of the parameter of W and b
# that's why we would use machining learning to get them

#trainingData = [1,2,3,4,5,6,7]
#trainingLabel = [2.5,4.7,7.2,9.5,10.8,13.1,14.8]

#trainingData = [1,2,3,4]
#trainingLabel = [0,-1,-2,-3]

trainingData = [1,2,3,4,5,6]
trainingLabel = [2.9,4.8,7.2,9.1,10.9,13.1]

def training():
    #x should be input data
    x = tf.placeholder(tf.float32)
    W = tf.Variable([1.9], tf.float32)
    b = tf.Variable([0.8], tf.float32)
    #y is calculated by the model
    y = W * x + b
    #y_ is the value of the label
    y_ = tf.placeholder(tf.float32)

    #init all the varable incluing W and b
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    squared_deltas = tf.square( y_ - y)
    loss = tf.reduce_sum(squared_deltas)

    print(sess.run([W, b]))
    print("sess.run(loss): ", sess.run(loss, {x: trainingData, y_: trainingLabel}))
    #exit(1)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    for i in range(100):
        sess.run(train, {x: trainingData, y_: trainingLabel})
        print("step {}:".format(i))
        print(sess.run([W, b]))
        print("sess.run(loss): ", sess.run(loss, {x: trainingData, y_: trainingLabel}))

    print("sess.run(loss): ", sess.run(loss, {x: trainingData, y_: trainingLabel}))
    print(sess.run([W, b]))
    #save the model for inference
    saver = tf.train.Saver()
    save_path = "./model/model.ckpt"
    saver.save(sess, save_path)
    #save for tensorflow board
    file_writer = tf.summary.FileWriter('.', sess.graph)

if __name__ == "__main__":
    training()