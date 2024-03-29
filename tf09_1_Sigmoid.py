import tensorflow as tf 
tf.set_random_seed(777) # for reproducibility

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]

y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

# Placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias') 

# hypothesis = tf.sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W))))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# hypothesis = WX+b
# hypothesis tf.matmul(WX+b) (0, 1)사이로 수렴해지기 위해서, tf.sigmoid로 씀

# cost/loss function Logistic Regression에서 cost는 log값 때문에 (-)가 붙음
# 로지스틱 회귀: 이진 분류
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        cost_val, _= sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report 
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis:", h, "\nCorrect (Y): ", c, "\nAccuracy:", a)

'''
model.predict, model.accuracy

'''
