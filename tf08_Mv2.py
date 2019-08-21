# Multi-variable linear regression 1
import tensorflow as tf 
tf.set_random_seed(777) # for reproducibility

x_data = [[73., 80., 75.,],
          [93., 88., 93.,],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# Placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias') # output = bias

hypothesis = tf.matmul(X, W) + b # matmul: 행렬의 곱셈

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize. Need a very small learning rate for this data set
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화

for step in range(2001):
    cost_val, hy_val, _= sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 10 == 0:
        print(step, "COST: ", cost_val, "\nPrediction:\n", hy_val)

'''
input_dim = 3
output_dim = 1

feed_dict={X: x_data, Y: y_data} => (x_data: None, 3), (y_data: None, 1)

'''