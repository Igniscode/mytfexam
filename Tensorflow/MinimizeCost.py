import tensorflow as tf
import matplotlib.pyplot as plt

#tf graph input
X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_sample = len(X)

# Set model weights
W = tf.placeholder(tf.float32)

# Construct a linear model
hypothesis = tf.multiply(X,W)

# Cost function
cost = tf.reduce_sum(tf.pow(hypothesis-Y,2))/(m)

#Initializing th ariables
init = tf.global_variables_initializer()

#for graphs
W_val = []
cost_val = []

#Launch the graph
sess = tf.Session()
sess.run(init)
for i in range(-30, 50):
    print (i*0.1, sess.run(cost, feed_dict={W: i*0.1}))
    W_val.append(i*0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

#Graphic display
plt.plot(W_val, cost_val,'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()