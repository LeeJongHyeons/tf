import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 옵션 설정
learning_rate = 0.001
total_epoch = 30
batch_size = 128

# 가로 픽셀수 n_input 으로, 세로 픽셀수 입력 단계인 n_step으로 설정
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

###########
# 신경망 모델 구성
###########
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

print(X) # (?, 28, 28)
print(Y) # (?, 10)
print(W) # (128, 10)
print(b) # (?, 28)

# RNN 에 학습에 사용할 셀을 생성
# 다음 함수들을 사용하면 다른 구조와 셀로 간단하게 변경
# BasicRNNCell, BasicLSTMCell, GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states, = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

print(outputs)

# 결과를 Y의 다음 형식과 바꿔야 하기 때문에
# Y: [batch_size, n_class]
# outputs의 형태를 이에 맞춰 변경
# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
#        -> [batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
print(outputs) # (28, ?, 128)
outputs = outputs[-1]
print(outputs) # (?, 128)

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

##########
# 신경망 모델 학습
##########
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # X 데이터를 RNN 입력 데이터 맞게 [batch_size, n_step, n_input] 형태로 변환
        batch_xs =  batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})

        total_cost += cost_val 

    print('Epoch :', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')


############################
# 결과 확인
############################

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))

'''
Epoch : 0001 Avg. cost = 0.518
Epoch : 0002 Avg. cost = 0.234
Epoch : 0003 Avg. cost = 0.182
Epoch : 0004 Avg. cost = 0.162
Epoch : 0005 Avg. cost = 0.142
Epoch : 0006 Avg. cost = 0.131
Epoch : 0007 Avg. cost = 0.116
Epoch : 0008 Avg. cost = 0.119
Epoch : 0009 Avg. cost = 0.107
Epoch : 0010 Avg. cost = 0.098
Epoch : 0011 Avg. cost = 0.099
Epoch : 0012 Avg. cost = 0.090
Epoch : 0013 Avg. cost = 0.092
Epoch : 0014 Avg. cost = 0.088
Epoch : 0015 Avg. cost = 0.085
Epoch : 0016 Avg. cost = 0.087
Epoch : 0017 Avg. cost = 0.076
Epoch : 0018 Avg. cost = 0.078
Epoch : 0019 Avg. cost = 0.081
Epoch : 0020 Avg. cost = 0.077
Epoch : 0021 Avg. cost = 0.074
Epoch : 0022 Avg. cost = 0.078
Epoch : 0023 Avg. cost = 0.070
Epoch : 0024 Avg. cost = 0.070
Epoch : 0025 Avg. cost = 0.074
Epoch : 0026 Avg. cost = 0.065
Epoch : 0027 Avg. cost = 0.069
Epoch : 0028 Avg. cost = 0.062
Epoch : 0029 Avg. cost = 0.063
Epoch : 0030 Avg. cost = 0.069
최적화 완료!
정확도: 0.9715

'''