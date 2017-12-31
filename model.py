import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

import os
import argparse
import json
import random

def create_model(x_train, y_train, x_test, y_test):

    def load_y_index_and_classes():
        with open("args.dat") as d:
            y_value = json.load(d)["y"]
        if y_value == "binary":
            return (4, 1)
        elif y_value == "binaryno23":
            return (5, 1)
        elif y_value == "linear":
            return (3, 5)
        else:
            return "Unknown predictor variable"
    data_dim = 1
    num_classes = load_y_index_and_classes()[1]

    # Expected input shape: (batch_size, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
    model = Sequential()
    branch = conditional({{choice(['two', 'three', 'four'])}})
    if branch == 'two':
        model.add(LSTM({{choice([8, 16, 32, 64, 128])}},
                       input_shape=(1, data_dim)))
    elif branch == 'three':
        model.add(LSTM({{choice([8, 16, 32, 64, 128])}}, return_sequences=True,
                       input_shape=(1, data_dim)))
        model.add(LSTM({{choice([8, 16, 32, 64, 128])}},
                       input_shape=(1, data_dim)))
    else:
        model.add(LSTM({{choice([8, 16, 32, 64, 128])}}, return_sequences=True,
                       input_shape=(1, data_dim)))
        model.add(LSTM({{choice([8, 16, 32, 64, 128])}}, return_sequences=True,
                       input_shape=(1, data_dim)))
        model.add(LSTM({{choice([8, 16, 32, 64, 128])}},
                       input_shape=(1, data_dim)))

    model.add({{choice([Dropout(0.5), Activation('linear')])}})
    model.add(Dense(num_classes, activation={{choice(['softmax', 'relu', 'tanh', 'sigmoid'])}}))

    # Change loss function based on input dimensionality
    if num_classes == 1:
        model.compile(loss={{choice(['mean_squared_error', 'mean_absolute_error',
                                 'binary_crossentropy'])}}, metrics=['binary_accuracy'],
                 optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
    else:
        model.compile(loss={{choice(['mean_squared_error', 'mean_absolute_error'])}}, metrics=['categorical_accuracy'],
                 optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

    model.fit(x_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=100,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])

    score, acc = model.evaluate(x_test, y_test, verbose=0)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def data():
    def load_x_indices():
        with open("args.dat") as d:
            x_value = json.load(d)["X"]
        if x_value == "fixations":
            return [0]
        elif x_value == "xy":
            return [1]
        elif x_value == "fixationsxy":
            return [0, 1]
        else:
            return "Unknown input variables"

    def load_y_index_and_classes():
        with open("args.dat") as d:
            y_value = json.load(d)["y"]
        if y_value == "binary":
            return (4, 2)
        elif y_value == "binaryno23":
            return (5, 2)
        elif y_value == "linear":
            return (3, 5)
        else:
            return "Unknown predictor variable"

    # Set the seed to achieve same sorting
    random.seed(6781693213)

    ids = list(range(0, 99))
    random.shuffle(ids)

    ids_train = ids[:80]
    ids_test = ids [80:]

    x_data_indices = load_x_indices()
    y_data_index = load_y_index_and_classes()[0]

    with open("modified_data.txt") as data:
        rows = json.load(data)

        # Remove NAN values if 3 is removed
        if y_data_index == 5:
            ids_train = [x for x in rows[y_data_index] if not x == None]
            ids_test = [x for x in rows[y_data_index] if not x == None]

        if len(x_data_indices) == 2:
            x_train = np.take(rows[x_data_indices[0]:x_data_indices[1]], ids_train).reshape(-1, 1, 1)
            x_test = np.take(rows[x_data_indices[0]:x_data_indices[1]], ids_test).reshape(-1, 1, 1)
            y_train = (np.take(rows[y_data_index], ids_train) - 1).reshape(-1, 1)
            y_test = (np.take(rows[y_data_index], ids_test) - 1).reshape(-1, 1)
        elif x_data_indices[0] == 1:
            x_train = np.take(rows[x_data_indices[0]], ids_train).reshape(-1, 1)
            x_test = np.take(rows[x_data_indices[0]], ids_test).reshape(-1, 1)
            y_train = (np.take(rows[y_data_index], ids_train) - 1).reshape(-1, 1)
            y_test = (np.take(rows[y_data_index], ids_test) - 1).reshape(-1, 1)
        else:
            x_train = np.take(rows[x_data_indices[0]], ids_train).reshape(-1, 1, 1)
            x_test = np.take(rows[x_data_indices[0]], ids_test).reshape(-1, 1, 1)
            y_train = (np.take(rows[y_data_index], ids_train) - 1).reshape(-1, 1)
            y_test = (np.take(rows[y_data_index], ids_test) - 1).reshape(-1, 1)



    return x_train, y_train, x_test, y_test

# In[16]:

if __name__ == '__main__':
    global x_data_indices, y_data_index, num_classes
    parser = argparse.ArgumentParser(description='Run facial recognition model.')
    parser.add_argument('X', metavar='X', type=str,
                       help='the input data; either fixations, xy or fixationsxy')
    parser.add_argument('y', metavar='y', type=str,
                       help='the output data; either binary, binaryno23, linear')

    args = parser.parse_args()

    with open("args.dat", "w") as d:
        json.dump({"X":args.X,"y":args.y}, d)

    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
print("Best run:")
print(best_run)
os.remove("args.dat")
