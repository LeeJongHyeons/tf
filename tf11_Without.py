import tensorflow as tf 
import numpy as np

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007], 
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973], 
                [816, 820.958984, 1008100, 815.48999, 819.23999], 
                [819.359985, 823, 1188100, 818.469971, 818.97998], 
                [819, 823, 1198100, 816, 820.450012], 
                [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]]) 

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# placeholder
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(101):
   cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
   print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

'''
0 Cost:  1784615300000.0
Prediction:
 [[ -941908.56]
 [-1895702.1 ]
 [-1491373.6 ]
 [-1045575.1 ]
 [-1232189.2 ]
 [-1242555.6 ]
 [-1138878.5 ]
 [-1449896.6 ]]
1 Cost:  1.9607214e+33
Prediction:
 [[3.1234838e+16]
 [6.2878931e+16]
 [4.9464584e+16]
 [3.4674411e+16]
 [4.0865647e+16]
 [4.1209606e+16]
 [3.7770028e+16]
 [4.8088756e+16]]
2 Cost:  inf
Prediction:
 [[-1.0353209e+27]
 [-2.0842072e+27]
 [-1.6395707e+27]
 [-1.1493303e+27]
 [-1.3545472e+27]
 [-1.3659482e+27]
 [-1.2519388e+27]
 [-1.5939669e+27]]

100 Cost:  nan
Prediction:
 [[nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]]
'''