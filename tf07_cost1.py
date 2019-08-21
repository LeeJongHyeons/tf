# Minimizing Cost
import tensorflow as tf 
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

w = tf.placeholder(tf.float32)
# ===================== compile ============================================
# Our hypothesis for linear model x * w 
# bios b를 생략, 즉 hypothesis(weight) = x * w
hypothesis = x * w

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))
# ==========================================================================

# Variables for plotting cost function
# cost(loss) 함수를 그래프로 출력하기 위한 변수설정
w_history = []
cost_history = []

with tf.Session() as sess:
    # w 범위 -30부터 50까지에 대하여  cost(loss) 함수의 값을 구하여 저장
    for i in range(-30, 50):
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict={w: curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)
# cost,weight
# Show the cost function
# matplotlib 모듈을 이용하여 cost(loss) 함수 그리기
plt.plot(w_history, cost_history)
plt.show()
