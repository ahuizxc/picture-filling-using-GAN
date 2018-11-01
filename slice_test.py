import tensorflow as tf 
import numpy
tt = numpy.array([[[1, 1, 1], [2, 2, 2], [7, 7, 7]], [[3, 3, 3], [4, 4, 4], [8, 8, 8]], [[5, 5, 5], [6, 6, 6], [9, 9, 9]]])
print(tt)
t = tf.constant([[[1, 1, 1], [2, 2, 2], [7, 7, 7]], [[3, 3, 3], [4, 4, 4], [8, 8, 8]], [[5, 5, 5], [6, 6, 6], [9, 9, 9]]]) 
z1 = tf.strided_slice(t, [1], [-1], [0]) 
z2 = tf.strided_slice(t, [1, 0], [-1, 2], [1, 1]) 
z3 = tf.strided_slice(t, [1, 0, 1], [-1, 2, 3], [1, 1, 1]) 
with tf.Session() as sess: 
    print('z: ',sess.run(t))
    print('z1: ', sess.run(z1)) 
    print('z2: ', sess.run(z2))
    print('z3: ', sess.run(z3))