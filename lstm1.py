import os
import time
import warnings
import numpy as np
import pandas as pd
from numpy import newaxis
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

#remove numpy and tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def get_data(filename, window_size, normalise_window):

    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    window_size = window_size + 1
    result = []

    # for loop to add data
    for i in range(len(data) - window_size):
        result.append(data[i: i + window_size])

    # normalize (convert to prices to daily returns)
    if normalise_window:
        result, x = normalise_windows(result)

    # convert it to an array
    result = np.array(result)
    x = np.array(x)

    # 90% training set
    row = round(0.9 * result.shape[0])
    # Taking all columns and 90% rows for training
    train = result[:int(row), :]

    # Each window is treated separately so successive windows are not sequential
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    # Keras LSTM works by taking a numpy array of 3 dimensions (N, W, F) where N is the number of training
    # samples, W is the window size & F is the number of features of each sequence
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test, x]


def normalise_windows(window_data):
    normalised_data = []
    arr = []
    for w in window_data:
        arr.append(w[0])
        normalised_window = [((float(p) / float(w[0])) - 1) for p in w]
        normalised_data.append(normalised_window)
    return normalised_data, arr

def denormalise_windows(normalised_window_data, actual):
    denormalised_data = []
    row = round(0.9 * actual.shape[0])
    for w in normalised_window_data:
        denormalised_window = [(float(actual[row]) * (float(w) + 1))]
        denormalised_data.append(denormalised_window)
        row = row + 1
    return denormalised_data


def build_model(layers):
    # input layer
    model = Sequential()

    # First LSTM layer (Hidden layer)
    model.add(LSTM(input_shape=(layers[1], layers[0]), output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.5))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.5))

    # Output layer with linear activation
    # 1 output class
    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    # Compile the model with rmsprop and mse
    # usually a good choice for RNN's and works for batch training
    start = time.time()

    rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, decay=0.05)
    model.compile(loss="mse", optimizer= rmsprop)

    # for daily preds
    model.compile(loss="mse", optimizer='adam')
    print("> Compilation Time : ", time.time() - start)
    return model


def predict_point_by_point(model, data):

    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):

    # Predict sequence of prediction_len steps before shifting prediction run forward by 50 steps
    prediction_seqs = []

    for i in range(int(len(data)/prediction_len)):
        # initial window frame
        curr_frame = data[i*prediction_len]
        predicted = []

        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            # dropping first test data element and appending prediction to the end
            curr_frame = curr_frame[1:]
            # inserts last value from predicted in curr_frame with window_size as index
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)

    return prediction_seqs
