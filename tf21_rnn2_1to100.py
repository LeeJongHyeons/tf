# 1 ~ 100까지의 숫자를 이용해서
# 6개씩 잘라서 rnn 구성
# train, test 분리

# 1,2,3,4,5,6 : 7
# 2,3,4,5,6,7 : 8
# 3,4,5,6,7,8 : 9
# ...
# 94,95,96,97,98,99 : 100

# predict : 101 ~ 110까지 예측
# 지표: RMSE

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

a = np.array(range(1,101))

size = 7
def split_7(seq, size):  
    aaa =[]
    for i in range(len(a)-size+1):
        subset = a[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_7(a, size)
print("==========================")
print(dataset)

x_train = dataset[:, 0:6]
y_train = dataset[:,6]

print(x_train.shape)  # (6,4)
print(y_train.shape)  # (6, )

x_train = np.reshape(x_train, (len(a)-size+1,4,1))

print(x_train.shape) # (6,4,1)

x_test = np.array([[[11],[12],[13],[14]],[[12],[13],[14],[15]],[[13],[14],[15],[16]],[[14],[15],[16],[17]]])
y_test = np.array([15,16,17,18])

print(x_test.shape)
print(y_test.shape)

x_test = np.array(([101, 102, 103, 104, 105, 106, 107, 108, 109, 110]))
print(x_test.shape)



# 모델
model = Sequential()
model.add(LSTM(20, input_shape=(6,1), return_sequences=True))
model.add(LSTM(10, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(10)) 

model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=1000, mode='auto')
model.fit(x_train, y_train, epochs=10000, batch_size=3, verbose=1, callbacks=[early_stopping])

loss, acc = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error  
def RMSE(y_test, y_predict): # 원래값, 예측값
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

print('loss :', loss)
print('acc :', acc)
print('y_predict(x_test) : \n', y_predict)
