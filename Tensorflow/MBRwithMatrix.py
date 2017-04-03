import tensorflow as tf

x_data = [[1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]
y_data = [1, 2, 3, 4, 5]

#try to find values for W and b thar comput y_data = W * x_data + b
#(We know that W should be 1 and b 0, but Tensorflow will
#figure that out for us.
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))

b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

#Our hypotesis
hypothesis = tf.matmul(W, x_data) + b

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
        print (step, sess.run(cost), sess.run(W), sess.run(b))