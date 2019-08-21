# Multi-variable linear regression 1
import tensorflow as tf 
tf.set_random_seed(777) # for reproducibility

x1_data = [77., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]
# Placeholders for a tensor that will be always fed
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2:x2_data,
                                              x3: x3_data, Y: y_data})

    if step % 10 == 0:
        print(step, "cost: ", cost_val, "\nPrediction:\n", hy_val)

'''
변수 3개를 넣어서, x1 * w1 + x2 * w2 + x3 * w3 + bias 구성
cost => const_val
hypothesis: 예측값
print(step(순서), cost_Val(loss), hy_val(hypothesis))

0 cost:  62333.887
Prediction:
 [-73.610954 -78.27629  -83.83015  -90.80437  -56.976486]
10 cost:  5.9403543
Prediction:
 [150.50902 186.47713 177.02235 193.2493  144.97726]

.
.
1990 cost:  2.8949604
Prediction:
 [151.80125 186.56703 178.2324  194.49445 144.56683]
2000 cost:  2.8862462
Prediction:
 [151.80383 186.5642  178.23428 194.49628 144.56264]

'''