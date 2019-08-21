import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost / loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
# Try to change learning rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initializes TensorFlow variables 
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        W_val, cost_val, _= sess.run([cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        
        print(step, cost_val, W_val)

    # predict
    print("prediction: ", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the acuuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))


'''
prediction:  [2 2 2]
learning-rate:0.1, Accuracy:  1.0

prediction:  [2 1 1]
learning-rate:0.05, Accuracy: 0.33333334

prediction:  [2 2 2]     
learning-rate:0.5, Accuracy:  1.0

prediction:  [2 2 2]
learning-rate:0.25, Accuracy:  1.0


prediction:  [2 2 2]
learning-rate:1, Accuracy:  1.0

'''