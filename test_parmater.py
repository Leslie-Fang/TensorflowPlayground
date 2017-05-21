import tensorflow as tf
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
sess = tf.Session()
print("sess.run(node3): ",sess.run(adder_node,{a: [1,3], b: [2, 4]}))
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))