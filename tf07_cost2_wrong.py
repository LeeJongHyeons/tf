import tensorflow as tf 
# tf Graph Input
x = [1, 2, 3]
y = [1, 2, 3]

# Set wrong model weights
w = tf.Variable(5.0) 

# Linear model
hypothesis = x * w #(weight=5)
# 결과: (5, 10, 15)
# ================== compile ===============================================
# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))
# Minimize: Gradient 
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# ==========================================================================

# Launch the graph in a session
with tf.Session() as sess:
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    for step in range(101): # 변수 초기화
        _, w_val = sess.run([train, w]) # train 결과 5를 주고, weight값이 1.0을 줌
        print(step, w_val)
# 잘못된 weight값을 주더라도 훈련시킨 과정을 통해, 좋은 결과를 나타날 수 있음
# input, ouput 늘어나는 형태!! 


