import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data
'''
tf.set_random_seed(77)

xy = np.loadtxt("D:/study/data/test0822LJH.csv", encoding="utf-8", nemes=['date', 'kp_0h', 'kp_3h', 'kp_6h', 'kp_12h', 'kp_15h', 'kp_18h', 'kp_21h'], delimiter=",", dtype=np.float32)
x = xy[:, 0:-1]
y = xy[:, [-1]]

print(x_data.shape, "\n", x_data, len(x_data))
print(y_data.shape, "\n", y_data)

from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = train_test_split(x, y, random_state=6, test_size=0.5)

x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

x_train = x_train / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

#print(x_test.shape)
#print(x_train.shape)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
OneHotEncoder = OneHotEncoder(categorical_features= [0])
y_test = OneHotEncoder.fit_transform(y_test).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
OneHotEncoder = OneHotEncoder(categorical_features= [0])
y_train = OneHotEncoder.fit_transform(y_train).toarray()

#print(y_train.shape)
#print(y_test.shape)

'''

mnist = input_data.read_data_sets("D:/study/data/test0822LJH.csv", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 16])
Y = tf,placeholder(tf.float32, [None, nb_classes])

#Y_one_hot = tf.one_hot(Y, nb_classes)
#Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes], name="weight"))
b = tf.Variable(tf.random_normal([nb_classes]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

#logits = tf.matmul(X, W) + b
#hypothesis = tf.nn.softmax(logits)

#cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
#cost = tf.reduce_mean(cost_i)

#train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#rediction = tf.argmax(hypothesis, 1)

#correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)               

        for step in range(2000):
             sess.run(train, feed_dict={X: x_data, Y: y_data})
                if step % 200 == 0:
                        loss, accu = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
                        print("step: {:5}\t cost: {:3f}\t accuracy: {:2%}".format(step, loss, accu))

        pred = sess.run(prediction, feed_dict={X: x_data, Y: y_data})

        for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.num_batch(batch_size)
                c, _= sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})

                avg_cost += c / total_batch
        print('Epoch:', '%04d', % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost)))
print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

r = random.randint(0, mnist.test.num_examples -1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction:", sess.run(tf.argamax(hypothesis, 1), feed_dict={X: mnist.test.images, Y: mnist.test.images[r: r+1]}))

'''
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1], name="weight"))
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.squre(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=le-5)
train = optimizer.minimize(cost)


predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32)) 

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(20001):
   #cost_, _= sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
    cost_val, hy_val, _= sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost :", cost_val, "\nPrediction :\n", hy_val)

print("Your Score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))
print("Other Scores will be", sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

from sklearn.metrics import mean_squared_error
def RMSE(y_data, hy_val):
    return np.sqrt(mean_squared_error(y_data, hy_val))
print("RMSE:", RMSE(y_data, hy_val))

'''
