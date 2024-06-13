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

# train_data = np.zeros(step*(num_lines-step+1),num_rows)
# # for i in range(step*(num_lines-step+1)):
# #     for j in range(num_rows):
# #         train_data[i,j] = i_data[i,j]
# # print(train_data)


from keras import utils

utils.to_categorical(1, num_classes=n)
nn = 0
n = 6
step = 7
a = np.zeros(7)
change_data = np.zeros((i_data.shape[0], 1))
for i in range(i_data.shape[0]):
    if 0 <= i_data[i, nn] <= 50:
        change_data[i] = 0

    if 50 < i_data[i, nn] <= 150:
        change_data[i] = 1
        a[0] += 1
    if 150 < i_data[i, nn] <= 250:
        change_data[i] = 2
        a[1] += 1
    if 250 < i_data[i, nn] <= 350:
        change_data[i] = 3
        a[2] += 1
    if 350 < i_data[i, nn] <= 420:
        change_data[i] = 4
        a[3] += 1
    if 420 < i_data[i, nn] <= 500:
        change_data[i] = 5
        a[4] += 1
    if 500 < i_data[i, nn] <= 600:
        change_data[i] = 6
        a[5] += 1
    if 600 < i_data[i, nn]:
        change_data[i] = 7
        a[6] += 1
print(change_data.shape)
print(a)
# plt.hist(change_data, rwidth=0.5, align='left', )
# plt.show()
train_data = np.zeros((3989*12, 6))
PM10_target = np.zeros((3989*12, 6))
train_data = []
train_target = []
for i in range(1, 3989):
    train_data.append(i_data[i - 1:i - 1 + step, :])
    train_target.append(utils.to_categorical(change_data[i - 1 + step], num_classes=n))
train_data = np.asarray(train_data)
train_data = train_data.reshape(3988, step * n)
train_target = np.asarray(train_target)
train_target = train_target.reshape(3988, n)

# val_data = np.zeros((1000, 12, 6))
# val_target = np.zeros((1000, 6))
val_data = []
val_target = []
for i in range(1, 989):
    val_data.append(i_data[i - 1 + 5000:i - 1 + 5000 + step, :])
    val_target.append(utils.to_categorical(change_data[i - 1 + 5000 + step], num_classes=n))
val_data = np.asarray(val_data)
val_data = val_data.reshape(988, step * n)
val_target = np.asarray(val_target)
val_target = val_target.reshape(988, n)
# test_data = np.zeros((989, 12, 6))
test_data = []
for i in range(1, 989):
    test_data.append(utils.to_categorical(change_data[i - 1 + 5000:i - 1 + 5000 + step], num_classes=n))
test_data = np.asarray(test_data)
test_data = test_data.reshape(988, step * n)

from keras.callbacks import ModelCheckpoint

filepath = 'last.best.hdf5'
# 有一次提升, 则覆盖一次.
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=2)
callbacks_list = [checkpoint]

from keras import layers
from keras import models

u=12
network = models.Sequential()
network.add(layers.Dense(u, activation='relu', input_shape=(6*step,), kernel_initializer="uniform"))
network.add(layers.Dense(u, activation='relu'))
network.add(layers.Dense(u, activation='relu'))
network.add(layers.Dense(u, activation='relu'))
# network.add(layers.Dense(u, activation='relu'))
# network.add(layers.Dense(u, activation='relu'))
network.add(layers.Dense(n, activation='softmax'))
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = network.fit(train_data, train_target, batch_size=50, epochs=300, verbose=2, callbacks=callbacks_list,
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
