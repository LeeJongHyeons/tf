import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def read_boston_data():
    boston = load_boston()
    feautures = np.array(boston, data)
    lables = np.array(boston.target)
    return feautures, lables

def normalizer(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return(dataset - mu) /sigma

def bias_vector(features, labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples), features], [n_training_samples, n_dim + 1])
    l = np.reshape(labels, [n_training_samples, 1])

    return f, l


features, labels = read_boston_data()
normalized_features = normalizer(features)
data, label = bias_vector(normalized_features, labels)
n_dim = data.shape[1]

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=100)

learning_rate = 0.01
training_epochs = 10000
log_loss = np.empty(shape=[1], dtype=float)
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.placeholder(tf.ones([n_dim, 1]))

y_ = tf.matmul(X, W)
cost_op = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)
for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={X: x_train, Y: y_train})
    log_loss = np.append(log_loss, sess.run(cost_op, feed_dict={X: x_train, Y: y_train}))


plt.plot(range(len(log_loss)), log_loss)
plt.axis([0, training_epochs, 0, np.max(log_loss)])
plt.show()

pred_y = sess.run(y_, feed_dict={X: x_test})
mse = tf.reduce_mean(tf.square(pred_y - y_test))
print("MSE: %.4f" % sess.run(mse))

