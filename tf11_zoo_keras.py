from keras.models import Sequential
from keras.layers import Dense
import numpy 
import pandas as pd
import keras
import sklearn.preprocessing import standardscaler

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt('D:/study/data/data-04-zoo.csv', delimiter=',')
# dataset = pd.read_csv('D:/study/data/data-04-zoo.csv')
#dataset.head()

data = pd.DataFrame(dataset)
print(data.head())

x_data = dataset[:, 0:8]
y_data = dataset[:, 8]

scaler = standardscaler()
x_data = scaler.fit_transform(x_data)

#print(x_data.shape)
#print(y_data.shape)

model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(241, activation='relu'))
model.add(Dense(132, activation='relu'))
model.add(Dense(114, activation='relu'))
model.add(Dense(102, activation='relu'))
model.add(Dense(91, activation='relu'))
model.add(Dense(46, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_data, y_data, epochs=350, batch_size=100, verbose=2)

scores = model.evaluate(x_data, y_data)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predict = model.predict(x_data)
predict = model.predict_classes(x_data)
#print(predict)

rounded = [round(x_data[0]) for x in predict]
#print(rounded)

for i in range(5):
    print('%s => %d (expend %d)' % (x_data[i].tolist(), predict[i], y_data[i]))
    
_, accuracy = model.evaluate(x, y)
#print('Acc: %2.f' % (accuracy * 100))
print("\n Test Accuracy %.4f"% (model.evaluate(x_test, y_test)[1]))

