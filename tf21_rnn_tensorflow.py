import numpy as np
import tensorflow as tf
tf.set_random_seed(777)

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]]]

y_data = [[1, 0, 2, 3, 3, 4]]

num_classes = 5
input_dim = 5
hidden_size = 5
batch_size = 1
sequence_length = 6
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) 
Y = tf.placeholder(tf.int32, [None, sequence_length]) 

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

y_data = tf.constant([[1, 1, 1]])

prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)

weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

y_data = tf.constant([[1, 1, 1]])


prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)
prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(logits=prediction1, targets=y_data, weights=weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(logits=prediction2, targets=y_data, weights=weights)

sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval(), "Loss2", sequence_loss.eval())


X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets= Y, weights=weights)
loss= tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        loss, _= sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss: ", loss, "Prediction: ", result, "True Y: ", y_data)
        # print(sess.run(weights))

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\nPrediction str : ", ''.join(result_str))

