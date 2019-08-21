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

    # Fit the line with new training data
    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run(
            [train, cost, w, b], 
            feed_dict={x: [1, 2, 3, 4, 5], y: [2.1, 3.1, 4.1, 5.1, 6.1]}
        )
        if step % 20 == 0: 
            print(step, cost_val, w_val, b_val)

    # Testing our model
    print(sess.run(hypothesis, feed_dict={x: [5]})) # model.predict(sess.run()) 값을 출력
    print(sess.run(hypothesis, feed_dict={x: [2.5]}))
    print(sess.run(hypothesis, feed_dict={x: [1.5, 3.5]}))

    '''
    y = wx+b
    {x: [1, 2, 3, 4, 5], y: [2.1, 3.1, 4.1, 5.1, 6.1]}
    -weight가 1
    -bios가 1.1
    2000번 Train 출력 결과, cost값이 낮아서, 좋은 모델~!

    [6.1009192]
    [3.5992656]
    [2.5986042  nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbn4.599927 ]

    '''
    