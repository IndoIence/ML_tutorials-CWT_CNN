import pywt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from load_data import read_signals, read_labels ,randomize_dataset
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import History
history = History()




INPUT_FOLDER_TEST = 'data/UCI HAR Dataset/test/Inertial Signals/'
INPUT_FOLDER_TRAIN = 'data/UCI HAR Dataset/train/Inertial Signals/'

INPUT_FILES_TRAIN = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                     'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                     'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

INPUT_FILES_TEST = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                    'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                    'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

LABELFILE_TRAIN = 'data/UCI HAR Dataset/train/y_train.txt'
LABELFILE_TEST = 'data/UCI HAR Dataset/test/y_test.txt'

activities_description = {
    1: 'walking',
    2: 'walking upstairs',
    3: 'walking downstairs',
    4: 'sitting',
    5: 'standing',
    6: 'laying'
}

train_signals, test_signals = [], []

for input_file in INPUT_FILES_TEST:
    signal = read_signals(INPUT_FOLDER_TEST + input_file)
    test_signals.append(signal)
test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
for input_file in INPUT_FILES_TRAIN:
    signal = read_signals(INPUT_FOLDER_TRAIN + input_file)
    train_signals.append(signal)
train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

train_labels = read_labels(LABELFILE_TRAIN)
test_labels = read_labels(LABELFILE_TEST)

[no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
[no_signals_test, no_steps_test, no_components_test] = np.shape(test_signals)
no_labels = len(np.unique(train_labels[:]))

print("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train,
                                                                                               no_steps_train,
                                                                                               no_components_train))
print("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test,
                                                                                              no_steps_test,
                                                                                              no_components_test))
print("The train dataset contains {} labels, with the following distribution:\n {}".format(np.shape(train_labels)[0],
                                                                                           Counter(train_labels[:])))
print("The test dataset contains {} labels, with the following distribution:\n {}".format(np.shape(test_labels)[0],
                                                                                          Counter(test_labels[:])))

uci_har_signals_train, uci_har_labels_train = randomize_dataset(train_signals, np.array(train_labels))
uci_har_signals_test, uci_har_labels_test = randomize_dataset(test_signals, np.array(test_labels))

######## Applying CWT to the signals and saving the scaleograms to an numpy ndarray

scales = range(1, 128)
waveletname = 'morl'
train_size = 5000
train_data_cwt = np.ndarray(shape=(train_size, 127, 127, 9))

for i in range(train_size):
    if i % 1000 == 0:
        print(i)
    for j in range(9):
        signal = uci_har_signals_train[i, :, j]
        coef, freq = pywt.cwt(signal, scales, waveletname, 1)
        coef_ = coef[:, :127]
        train_data_cwt[i, :, :, j] = coef_

test_size = 500
test_data_cwt = np.ndarray(shape=(test_size, 127, 127, 9))
for i in range(test_size):
    if i % 100 == 0:
        print(i)
    for j in range(9):
        signal = uci_har_signals_test[i, :, j]
        coef, freq = pywt.cwt(signal, scales, waveletname, 1)
        coef_ = coef[:, :127]
        test_data_cwt[i, :, :, j] = coef_

np.save('data/scaleograms/test_data_cwt.npy', test_data_cwt)

np.save('data/scaleograms/train_data_cwt.npy', train_data_cwt)

x_train = train_data_cwt
y_train = list(uci_har_labels_train[:train_size])
x_test = test_data_cwt
y_test = list(uci_har_labels_test[:test_size])


img_x = 127
img_y = 127
img_z = 9
input_shape = (img_x, img_y, img_z)

num_classes = 6
batch_size = 16
num_classes = 7
epochs = 10

x_train


train_data_cwt = np.load('data/scaleograms/train_data_cwt.npy')
test_data_cwt = np.load('data/scaleograms/test_data_cwt.npy')

x_train = train_data_cwt
y_train = list(uci_har_labels_train[:train_size])
x_test = test_data_cwt
y_test = list(uci_har_labels_test[:test_size])

img_x = 127
img_y = 127
img_z = 9
input_shape = (img_x, img_y, img_z)

num_classes = 6
batch_size = 16
num_classes = 6
epochs = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=epochs, verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))

