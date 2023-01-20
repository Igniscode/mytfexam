########## Caution ###########
# Because of difference between python3 and python2
# some code was changed randomly 
import tensorflow as tf 

x_data = [1, 2, 3]
y_data = [1, 2, 3]

#try to find values for W and b thar comput y_data = W * x_data + b
#(We know that W should be 1 and b 0, but Tensorflow will
#figure that out for us.
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Our hypothesis
hypothesis = W * x_data + b

#Simplified cost function (get average)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

#Minimize
a = tf.Variable(0.1) # Learnig rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Before starting, initialize the variables. We will 'run' this first.
########## code was modified ########### 
init = tf.global_variables_initializer()

#Launch the grape.
sess = tf.Session()
sess.run(init)

#Fit the line.
########## code was modified ###########
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(cost), sess.run(W), sess.run(b))