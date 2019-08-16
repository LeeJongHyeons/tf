import tensorflow as tf
tf.set_random_seed(777)
# seed 값으로 777사용, 난수 표에서 777에 매칭한 내용으로 값이 원할하게 지정

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

#Variable  기존 변수와는 다른 tensor에서 사용하는 변수 trainable한 값

# w와 b값에 랜덤값으로 넣어줌
W=tf.Variable(tf.random_normal([1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')


# 선형 함수
hypothesis = x_train * W + b

# cost/loss funcion
# 만든 선형함수와 주어진 데이터들과 차이값 +- 차이가 있기 때문에, 제곱으로 값을 비교
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# optimizer, minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:# 여러번 하면서 20번마다 값을 출력
        print(step,sess.run(cost),sess.run(W),sess.run(b))

'''
0 11.843552 [-0.00328879] [-1.3359501]
20 0.17985165 [1.1873045] [-0.77016306]
40 0.06744872 [1.2873796] [-0.68607277]
60 0.06038854 [1.2842411] [-0.6492692]

.
.

1960 6.4399537e-06 [1.0029473] [-0.0067001]
1980 5.8486967e-06 [1.0028089] [-0.00638524]
2000 5.3121207e-06 [1.002677] [-0.00608517]


'''

