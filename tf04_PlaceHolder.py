import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) #Constant: 상수, 노드의 형태를 출력
node2 = tf.constant(4.0) # 그래프의 형태를 출력!
node3 = tf.add(node1, node2) # Tensor add


# node 1 = Tensorflow 상수 값을 실수형의 3.0으로 지점
# node 2 = Tensorflow 상수 값을 4.0으로 지점
# node 3 = node1 + node2

print("node1:", node1, "node2:", node2)
print("node3:", node3)

sess = tf.Session()
#print("sess.run(node1, node2):", sess.run([node1, node2]))
#print("sess.run(node3):", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b   # + provides ad shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, feed_dict={a: 3, b:4.5}))
'''
7.5
[3. 7.]
22.5
placeholder: sess.run할때, feed_dict 함수를 넣음
adder_node: 변환하는 노드
feed_dict(a에 3을, b에 4.5를 넣음)

sess.run에 연산한 그래프를 집어넣어, 

'''