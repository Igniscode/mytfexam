import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

xy = np.loadtxt('trainer.txt', unpack = True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1,3], -2.0, 2.0))

hypothesis = tf.matmul(W , X)

cost = tf.reduce_mean(tf.square(hypothesis-Y))

a =tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(20001):
    if step % 100 == 0:
        print (step,' cost = ', sess.run(cost,feed_dict={X:x_data, Y:y_data}),' W = ', sess.run(W))
    sess.run(train , feed_dict={X:x_data , Y:y_data})


print (sess.run(hypothesis, feed_dict={X:[[1.] ,[2.] ,[4.]]}))
print (sess.run(hypothesis, feed_dict={X:[[1.] ,[1.] ,[5.]]}))





