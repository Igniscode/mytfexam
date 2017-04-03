import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack = True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

print('x',x_data)
print('y',y_data)

#try to find values for W and b thar comput y_data = W * x_data + b
#(We know that W should be 1 and b 0, but Tensorflow will
#figure that out for us.
W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))

#Our hypotesis
#previos hypotesis with b, hypothesis = tf.matmul(W, x_data) + b
hypothesis = tf.matmul(W, x_data)

#SImplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#Minimize
a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Befor starting, initallize the variables, We will 'run' this first.
init = tf.global_variables_initializer()

#Launch the graph.
sess = tf.Session()
sess.run(init)

#fit the line.
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W))