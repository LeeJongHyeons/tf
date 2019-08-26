import numpy as np
import tensorflow as tf

# 1. 데이터 구축

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)

print(_data) # (7, 1)
print(_data.shape)
print(type(_data))
print(_data.dtype)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')

'''
알파벳 순서대로 OneHotEncoder하는데, 'float32' 형식으로 순서대로 출력 (float64 -> float32 바뀜)

'''
print(_data) # (7, 5)
print(_data.shape)
print(type(_data))
print(_data.dtype)

x_data = _data[:6,] # (6, 5)
y_data = _data[1:,] # (6, 5)
y_data = np.argmax(y_data, axis=1)

x_data = x_data.reshape(1, 6, 5) # (1, 6, 5)
y_data = y_data.reshape(1, 6)

print(x_data.shape) # (1, 6, 5)
print(x_data.dtype) 
print(y_data.shape) # (6, )

# 데이터 구성
# x: (batch_size, sequence_length, input_dim) 1,6,5
# 첫번째 아웃풋: hidden_size = 2
# 첫번째 결과: 1,6,5

num_classes = 5
batch_size = 1        #(전체행)
sequence_length = 6   # 컬럼
input_dim = 5         # 몇개씩 작업
hidden_size = 5       # 첫번째 노드 출력 갯수
learning_rate = 0.1

# X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # (?, 6, 5)
# Y = tf.placeholder(tf.int32, [None, sequence_length]) # (?, 6)

X = tf.compat.v1.placeholder(tf.float32, [None, sequence_length, input_dim]) # (?, 6, 5)
Y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length]) # (?, 6)
print(X)
print(Y)

# 2. 모델 구성
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size) # 출력사이즈 
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

print(outputs)
print(outputs.shape) # (1, 6, 5)

# FC layer
# X_for_fc = tf.reshape(outputs, [-1, hidden_size]) # (6, 5)
# print(X_for_fc)
# outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

################################
# W, loss, train, prediction
################################

weights = tf.ones([batch_size, sequence_length]) # 임의로 1을 넣음

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction= tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(300):
        loss2, _= sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(i, "loss: ", loss2, "Prediction: ", result, "True Y: ", y_data)
        # print(sess.run(weights))

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str : ", ''.join(result_str))
        # range(100): Prediction str :  ihello
        # range(300): Prediction str :  ihello
        # range(400): Prediction str :  ihello

'''
0 loss: 1.71584 prediction:  [[2 2 2 3 3 2]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  eeelle

1 loss: 1.56447 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  llllll

2 loss: 1.46284 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  llllll

3 loss: 1.38073 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  llllll

4 loss: 1.30603 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  llllll

5 loss: 1.21498 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  llllll

6 loss: 1.1029 prediction:  [[3 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  lhlllo

7 loss: 0.982386 prediction:  [[1 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  ihlllo

8 loss: 0.871259 prediction:  [[1 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  ihlllo

9 loss: 0.774338 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  ihello

10 loss: 0.676005 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]

	Prediction str:  ihello

'''
