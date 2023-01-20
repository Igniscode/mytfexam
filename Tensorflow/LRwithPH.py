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

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis
hypothesis = W * X + b

#Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
a = tf.Variable(0.1) # Learnig rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

#Before starting, initialize the variables. We will 'run' this first.
########## This code was modified ########### 
init = tf.global_variables_initializer()

#Launch the grape.
sess = tf.Session()
sess.run(init)

#Fit the line.
########## This code was modified ###########
for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0: 
        print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

#Learns best fit it W: [1}, b: [0]
########## This code was modified ###########
print (sess.run(hypothesis, feed_dict={X:5}))
print (sess.run(hypothesis, feed_dict={X:2.5}))

