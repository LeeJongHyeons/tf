import tensorflow as tf

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# model parameters
w = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 모델 구성
hypothesis = x * w  +  b 

# ================  model.compile  ======================================
# cost / loss function
cost = tf.reduce_sum(tf.square(hypothesis - y)) # sum of the squares

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
# ======================================================================

# training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    w_val, b_val, cost_val = sess.run([w, b, cost], feed_dict={x: x_train, y: y_train})
    print(f"w: {w_val} b: {b_val} cost: {cost_val}")

'''   
-weight가 -1
-bios가 1
input_shape = (1, )
input_dim = (1)
cost값이 낮아서 좋은 모델~!

출력결과: model.add(1, Dense(1)) 
w: [-0.9999969] b: [0.9999908] cost: 5.699973826267524e-11

'''

