'''
Keras에서는 default 값으로 사용
tensorflow에서는 Learning-Rate 값을 명시
y = Wx+b은 최소의 loss(또는 cost) train 과정에서 찾아냄
 mse를 기준으로 하게되면 cost의 형태가 이차식의 형태를 가짐  
 최소의 loss, cost를 구해서 최적의 weight 값을 얻음 
 기울기의 자르는 크기에 따라서, 기울기의 크기가 달라짐(loss값에 영향을 미침)
 극점 기준으로 좌측의 경우는 과소적합 / 우측의 경우는 과대적합
 
 경사 하강도(Gradient Descent): 오차함수의 낮은 지점을 찾아가는 최적화 방법, 낮은 쪽의 방향을 찾기 위해 오차함수를 현재 위치에서 미분
 경사 하강도의 특징: 함수의 최저점을 구하기 좋은 방법으로 신경망과 같이 계산해야 하는 양이 많고, 접근하기 복잡한 수식에 잘 작동, 데이터가 불안전해도 유도리 있게 작동

'''
# 데이터 구성
import numpy as np  
x = np.array([1,2,3])
y = np.array([1,2,3])

# 모델 구성 
from keras.models import Sequential 
from keras.layers import Dense  
model = Sequential()

model.add(Dense(5, input_dim =1, activation = 'relu')) 
model.add(Dense(3))   
model.add(Dense(4))   
model.add(Dense(1))   

# 훈련
from keras.optimizers import Adam#(Adam: 확률 최적화), SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam 
optimizer = Adam(lr=0.000096)
model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
#model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x, y, epochs=100, batch_size=1) 

# 평가 예측
mse, _ = model.evaluate(x, y, batch_size=1)
print("mse :", mse)
 
ped1 = model.predict([1.5, 2.5, 3.5])
print(ped1)

#mse : 3.789561257387201e-14
#[[1.5]
# [2.5]
#  [3.5]]