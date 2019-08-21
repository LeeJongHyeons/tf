import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("Data/MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
# =================== layer 출력 ==============================================
W1 = tf.Variable(tf.random_normal([784, 100]))
b1 = tf.Variable(tf.random_normal([100]))
layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([100, 50]))
b2 = tf.Variable(tf.random_normal([50]))
layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([50, 48]))
b3 = tf.Variable(tf.random_normal([48]))
layer3 = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([48, 46]))
b4 = tf.Variable(tf.random_normal([46]))
layer4 = tf.nn.softmax(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([46, 74]))
b5 = tf.Variable(tf.random_normal([74]))
layer5 = tf.nn.softmax(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_normal([74, 72]))
b6 = tf.Variable(tf.random_normal([72]))
layer6 = tf.nn.softmax(tf.matmul(layer5, W6) + b6)

W7 = tf.Variable(tf.random_normal([72, 68]))
b7 = tf.Variable(tf.random_normal([68]))
layer7 = tf.nn.softmax(tf.matmul(layer6, W7) + b7)

W8 = tf.Variable(tf.random_normal([68, 44]))
b8 = tf.Variable(tf.random_normal([44]))
layer8 = tf.nn.softmax(tf.matmul(layer7, W8) + b8)

W9 = tf.Variable(tf.random_normal([44, 21]))
b9 = tf.Variable(tf.random_normal([21]))
layer9 = tf.nn.softmax(tf.matmul(layer8, W9) + b9)

W10 = tf.Variable(tf.random_normal([21,nb_classes]))
b10 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(layer9, W10) + b10)
# ===============================================================================================

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs): # 550
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
           batch_xs, batch_ys = mnist.train.next_batch(batch_size)
           c, _= sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})

           avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
    
    # sample image show and prediction
    r = random.randint(0, mnist.test.num_examples -1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))


    print("sample image shape:", mnist.test.images[r:r+1].shape)
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()