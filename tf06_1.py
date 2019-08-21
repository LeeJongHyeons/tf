# Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility = 랜덤값 고정

# w, b값을 랜덤값으로 넣음
w = tf.Variable(tf.random_normal([1]), name="weight") 
b = tf.Variable(tf.random_normal([1]), name="bias")

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

hypothesis = x * w  +  b  # Our hypothesis xw+b(우리가 원하는 선형함수)

# ================== compile ===============================================
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y)) # 만든 선형함수와 주어진 데이터들과 차이값 +- 차이가 있기 때문에 제곱으로 값을 비교
# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# =========================================================================

# Launch the graph in a session
with tf.Session() as sess:
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer()) # 변수 초기화

    # Fit the line: 2000번 반복시킴 => 20번에 한 번씩 현재 값들을 출력
    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run(
            [train, cost, w, b], feed_dict={x: [1, 2, 3], y: [1, 2, 3]}
        )
        if step % 20 == 0: 
            print(step, cost_val, w_val, b_val)

    # Testing our model
    print(sess.run(hypothesis, feed_dict={x: [5]})) # model.predict(sess.run()) 값을 출력
    print(sess.run(hypothesis, feed_dict={x: [2.5]}))
    print(sess.run(hypothesis, feed_dict={x: [1.5, 3.5]}))
    '''
    [5.0110054]
    [2.500915]
    [1.4968792 3.5049512]
    '''
    # Learns best fit w: [ 1. ], b:[ 0]
    