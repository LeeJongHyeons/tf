# Softmax Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777) # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5], 
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
# 4개의 속성을 가지고 있음
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
# one-hot encoding이 이미 이루어짐
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3 # class의 개수를 의미

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
# score를 logit이라고도 부른다고 합니다. tf.nn.softmax를 사용하지 않고도 간단히 구현가능
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy cost/loss 함수 구현
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# 복잡한 미분을 직접 계산할 필요없이, tensorflow가 알아서 계산
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val= sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        print(step, cost_val)
    print('--------------------------')
    
    # Testing & one-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    # hypothesis에 새로운 값을 넘겨줌
    print(a, sess.run(tf.argmax(a, 1)))
    # argmax는 두번째 인자로 넘겨준 1차원의 argument중에 가장 큰 값을 반환
    
    print('--------------------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(b, 1)))
    
    print('--------------------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))
    # 한 번에 여러개의 값을 확인해 볼 수 있음

'''
[[1.3890490e-03 9.9860185e-01 9.0612921e-06]] [1]
--------------------------
[[0.9311919  0.06290223 0.00590592]] [0]
--------------------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
--------------------------
[[1.3890478e-03 9.9860197e-01 9.0612930e-06]
 [9.3119192e-01 6.2902197e-02 5.9059118e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2] 

'''