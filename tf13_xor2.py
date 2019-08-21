import tensorflow as tf 
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 100]), name='weight')
b1 = tf.Variable(tf.random_normal([100]), name='bias')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([100, 100]), name='weight')
b2 = tf.Variable(tf.random_normal([100]), name='bias')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)


W3 = tf.Variable(tf.random_normal([100, 100]), name='weight')
b3 = tf.Variable(tf.random_normal([100]), name='bias')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W6 = tf.Variable(tf.random_normal([100, 1]), name='weight')
b6 = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.sigmoid(tf.matmul(layer3, W6) + b6)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, w_val = sess.run([cost, W2, train], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0:
            print(step, cost_val, w_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})

    print("\nHypothesis", h, "\nCorrect: ", c, "\nAccuracy: ", a)

    '''
    Hypothesis [[3.1101704e-04]
 [9.9957144e-01]
 [9.9953449e-01]
 [5.5900216e-04]]
Correct:  [[0.]
 [1.]
 [1.]
 [0.]]
Accuracy:  1.0

    '''