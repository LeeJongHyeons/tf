import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
                               # [5, 5, 1(흑백), 10]
# L1 ImgIn shape (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
print('W1 :', W1)
# W1 : <tf.Variable 'Variable:0' shape=(3, 3, 1, 32) dtype=float32_ref>

# kernel_size = [3, 3]
# 32 => output
# Conv  -> (?, 28, 28, 32)  28 -3 + 1 = 26
# pool  -> (?, 14, 14, 32)  

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') # Strides: 가로세로 똑같이 1개씩 움직임, 1칸씩 잘라서 작업
print('L1:', L1)
#L1: Tensor("Conv2D:0", shape=(?, 3, 28, 32), dtype=float32)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('L1:', L1)
#L1: Tensor("MaxPool:0", shape=(?, 2, 14, 32), dtype=float32)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# Conv  -> (?, 14, 14, 64) 
# pool  -> (?, 7, 7, 64)  
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') # [14, 14, 64]
print('L2:', L2)
# L2: Tensor("Conv2D_1:0", shape=(?, 2, 14, 64), dtype=float32)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print('L2:', L2)
# L2: Tensor("MaxPool_1:0", shape=(?, 1, 7, 64), dtype=float32)

############################### 추가 ###############################
L3 = tf.layers.conv2d(L2, 64, [3, 3], activation=tf.nn.relu)
L3 = tf.layers.max_pooling2d(L3, [2, 2], [2, 2])
L3 = tf.layers.dropout(L3, 0.7)

L4 = tf.contrib.layers.flatten(L3)
L4 = tf.layers.dense(L4, 256, activation=tf.nn.relu)
L4 = tf.layers.dropout(L4, 0.5)

L2_flat = tf.layers.dense(L4, 10, activation=None)
############################### 추가 ###############################
#L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64]) # [-1: 7, 7, 64를 reshape시킴]

############ 8/ 23 시작 ####################################
# Final FC 7x7x64 inputs -> 10 outputs
#W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
#b = tf.Variable(tf.random_normal([10]))
#logits = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
  
    for i in range(total_batch):
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      feed_dict = {X: batch_xs, Y: batch_ys}
      c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
      avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')