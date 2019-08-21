import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(77)

data = np.load("D:/study/data/iris2.csv")

x_data = data[:, 0:-1]
y_data = data[:, [-1]]

print(x_data.shape) # (150, 4)
print(y_data.shape) # (150, 1)

y_one_hot = tf.one_hot(y_data, 3)
y_one_hot = tf.reshape(y_one_hot, [-1, 3])
print(y_one_hot.shape) # (150, 1, 3)

x = tf.placeholder('float', [None, 4])
y = tf.placeholder('float', [None, 1])

'''
from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(x_data, y_data, batch_size=0.2)

print(x_train.shape) (120, 4)
print(y_train.shape) (120, 1)

'''
# layer1
W1 = tf.get_Variable('W1', [4, 50], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_uniform([50]))
L1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# layer2
W2 = tf.get_Variable('W2', [50, 25], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_uniform([25]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

# layer3
W3 = tf.get_Variable('W3', [25, 3], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_uniform([3]))
hypothesis = tf.nn.softmax(tf.matmul(L2, W3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(hypothesis, 1e-10, 1)), axis=1))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.01).minimize(cost)

predicted = tf.argmax(hypothesis, 1)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs = 50
batch_size = 32
num_iteration = int(x_data.shape[0] / batch_size)

with tf.Session() as sess:
    sess.run(tf.gloabl_variables_initializer())

    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iteration):
            batch_x = x_data[(i * batch_size): (1 * batch_size) + batch_size]
            batch_y = y_data[(i * batch_size): (1 * batch_size) + batch_size]

            _, cost_val = sess.run([optimize, cost], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += cost_val / num_iteration

            print('Epoch : {:04d}, cost: {:.9f}'.format(epoch+1, avg_cost))


    print('Learning Finished')

    print('Accuracy', sess.run(accuracy, feed_dict={X:x_data, Y: y_data}))

    print('Predict', sess.run(predicted, feed_dict={X:x_data}))
