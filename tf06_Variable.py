# 랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력

import tensorflow as tf
# tf.set_random_seed(777) = 랜덤값 고정

w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

print(w)

w = tf.Variable([0.3], tf.float32) 

'''
sess = tf.Session()     # w = 0.3
sess.run(tf.global_variables_initializer())
aaa = sess.run(w)
print(aaa)
sess.close()
'''

'''
sess = tf.InteractiveSession()  # w = 0.3  
sess.run(tf.global_variables_initializer())
aaa = w.eval()
print(aaa)
sess.close()
'''

sess = tf.Session()   # w = 0.3
sess.run(tf.global_variables_initializer())
aaa = w.eval(session=sess)
print(aaa)
sess.close()



