import numpy as np
# import os
import matplotlib.pyplot as plt
import xlrd

data = xlrd.open_workbook('2018站点空气质量指数实时报(审核).xls')
table = data.sheet_by_name('Sheet0')
num_lines = table.nrows
print(num_lines)
num_rows = table.ncols
print(num_rows)

i_data = []
for i in range(num_lines):
    for j in range(num_rows):
        i_data.append(table.cell(i, j).value)
i_data = np.asarray(i_data)
i_data = i_data.reshape(num_lines, num_rows)
print(i_data)
print(i_data.shape)

# step=12
# train_data = np.zeros(step*(num_lines-step+1),num_rows)
# # for i in range(step*(num_lines-step+1)):
# #     for j in range(num_rows):
# #         train_data[i,j] = i_data[i,j]
# # print(train_data)


from keras import utils

nn = 0
n = 6
change_data = np.zeros((i_data.shape[0], 1))
for i in range(i_data.shape[0]):
    if 0 <= i_data[i, nn] <= 50:
        change_data[i] = 0
    if 50 < i_data[i, nn] <= 75:
        change_data[i] = 1
    if 75 < i_data[i, nn] <= 100:
        change_data[i] = 2
    if 100 < i_data[i, nn] <= 150:
        change_data[i] = 3
    if 150 < i_data[i, nn] <= 200:
        change_data[i] = 4
    if i_data[i, nn] > 200:
        change_data[i] = 5
print(change_data)
from keras import utils
from keras.utils import np_utils

train_data = np.zeros((3989, 12, 6))
PM10_target = np.zeros((3989, n))
for i in range(1, 3990):
    train_data[i - 1] = i_data[i - 1:i - 1 + 12]
    PM10_target[i - 1] = np_utils.to_categorical(change_data[i - 1 + 12], num_classes=n)
print(PM10_target.shape)
val_data = np.zeros((1000, 12, 6))
val_target = np.zeros((1000, n))
for i in range(1, 990):
    val_data[i - 1] = i_data[i - 1 + 5000:i - 1 + 5000 + 12]
    val_target[i - 1] = np_utils.to_categorical(change_data[i - 1 + 5000 + 12], num_classes=n)

test_data = np.zeros((989, 12, 6))
for i in range(1, 990):
    test_data[i - 1] = i_data[i - 1 + 5000:i - 1 + 5000 + 12]

from keras.callbacks import ModelCheckpoint

filepath = 'PM2.5.best.hdf5'
# 有一次提升, 则覆盖一次.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=2)
callbacks_list = [checkpoint]
from keras import layers
from keras import models

network1 = models.Sequential()
# network1.add(layers.LSTM(70, input_shape=(7, 1), return_sequences=True, dropout=0.1, recurrent_dropout=0.2))
# network1.add(layers.LSTM(70, return_sequences=True))
network1.add(layers.LSTM(24, input_shape=(12, 6), return_sequences=True, dropout=0.1, recurrent_dropout=0.3))
network1.add(layers.LSTM(24, return_sequences=True, dropout=0.2))
network1.add(layers.LSTM(24))
network1.add(layers.Dense(24, activation='relu'))
network1.add(layers.Dense(24, activation='relu'))
network1.add(layers.Dense(n, activation='softmax'))
network1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = network1.fit(train_data, PM10_target, batch_size=128, epochs=300, verbose=2, callbacks=callbacks_list,
                       validation_data=(val_data, val_target))

loss = history.history['loss']
acc = history.history['acc']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Training loss and acc')
plt.legend()

val_loss = history.history['val_loss']
val_acc = history.history['val_acc']
epochs = range(1, len(val_loss) + 1)
plt.figure()
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Validation loss and acc')
plt.legend()
plt.show()

network1.load_weights('PM2.5.best.hdf5'.encode('utf-8').decode('utf-8'))
p = network1.predict_classes(test_data, batch_size=128)
h = 0
for i in range(1, 990):
    if p[i - 1] == change_data[i - 1 + 5000 + 12]:
        h = h + 1
predict_acc = h / 989
print('PM2.5 acc: %f' % predict_acc)
