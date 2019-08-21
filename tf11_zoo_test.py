import tensorflow as tf 
import numpy as np
tf.set_random_seed(777) # for reproducibility

xy = np.loadtxt('D:/study/data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7 # 0 ~ 6

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes) # one_hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)

'''

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias') 

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim
# rmrer,)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
#hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# =================== Cross entropy cost / loss =======================================================================
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
'''
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
'''
# =====================================================================================================================

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        acc_val, cost_val, _= sess.run([cost, accuracy, optimizer], feed_dict={X: x_data, Y: y_data})
        
        if step % 100 == 0:
            print("step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))
    
    # Let's see if we can predict
    pred = sess.run(pediction, feed_dict={X: x_data})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))


'''
step:     0     Cost: 0.376     Acc: 547.96%
step:   100     Cost: 0.792     Acc: 80.58%
step:   200     Cost: 0.881     Acc: 48.82%
step:   300     Cost: 0.901     Acc: 35.04%
step:   400     Cost: 0.941     Acc: 27.23%
step:   500     Cost: 0.950     Acc: 22.23%
step:   600     Cost: 0.970     Acc: 18.72%
step:   700     Cost: 0.970     Acc: 16.10%
step:   800     Cost: 0.970     Acc: 14.06%
step:   900     Cost: 0.970     Acc: 12.44%
step:  1000     Cost: 0.970     Acc: 11.13%
step:  1100     Cost: 0.990     Acc: 10.06%
step:  1200     Cost: 1.000     Acc: 9.17%
step:  1300     Cost: 1.000     Acc: 8.42%
step:  1400     Cost: 1.000     Acc: 7.79%
step:  1500     Cost: 1.000     Acc: 7.25%
step:  1600     Cost: 1.000     Acc: 6.78%
step:  1700     Cost: 1.000     Acc: 6.36%
step:  1800     Cost: 1.000     Acc: 6.00%
step:  1900     Cost: 1.000     Acc: 5.68%
step:  2000     Cost: 1.000     Acc: 5.39%

'''


