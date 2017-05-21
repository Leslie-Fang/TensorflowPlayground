import numpy as np
import tensorflow as tf
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

print("sess.run(node3): ",sess.run(loss ,{x:x_train, y:y_train}))

'''
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))'''

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
#sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:x_train, y:y_train})

print(sess.run([W, b]))
file_writer = tf.summary.FileWriter('.', sess.graph)