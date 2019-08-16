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
print("sess.run(node1, node2):", sess.run([node1, node2]))
print("sess.run(node3):", sess.run(node3))


'''
node1: Tensor("Const:0", shape=(), dtype=float32) node2: Tensor("Const_1:0", shape=(), dtype=float32)
node3: Tensor("Add:0", shape=(), dtype=float32)

sess.run(node1, node2): [3.0, 4.0]
sess.run(node3): 7.0

Print(node행) 하게되면 Node 형태 정보를 출력
Tensor 값 저장 및 연산은 기계에서 이해하는 방식으로 이용
Tensorflow에 적용하는 Session을 통해 해석

'''