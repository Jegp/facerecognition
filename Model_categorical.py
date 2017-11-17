import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

def create_model(x_train, y_train, x_test, y_test):
    data_dim = 1
    num_classes = 5

    # Expected input shape: (batch_size, data_dim)
    # Note that we have to provide the full batch_input_shape since the network is stateful.
    # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
    model = Sequential()
    branch = conditional({{choice(['two', 'three', 'four'])}})
    if branch == 'two':
        model.add(LSTM({{choice([8, 16, 32, 64, 128, 256, 512])}},
                       input_shape=(1, data_dim)))
    elif branch == 'three2':
        model.add(LSTM({{choice([8, 16, 32, 64, 128, 256, 512])}}, return_sequences=True,
                       input_shape=(1, data_dim)))
        model.add(LSTM({{choice([8, 16, 32, 64, 128, 256, 512])}},
                       input_shape=(1, data_dim)))
    else:
        model.add(LSTM({{choice([8, 16, 32, 64, 128, 256, 512])}}, return_sequences=True,
                       input_shape=(1, data_dim)))
        model.add(LSTM({{choice([8, 16, 32, 64, 128, 256, 512])}}, return_sequences=True,
                       input_shape=(1, data_dim)))
        model.add(LSTM({{choice([8, 16, 32, 64, 128, 256, 512])}},
                       input_shape=(1, data_dim)))

    model.add({{choice([Dropout(0.5), Activation('linear')])}})
    model.add(Dense(num_classes, activation={{choice(['softmax', 'relu', 'tanh', 'sigmoid'])}}))

    model.compile(loss={{choice(['categorical_crossentropy', 'mean_squared_error', 'mean_absolute_error'])}}, metrics=['categorical_accuracy'],
                 optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

    model.fit(x_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=100,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])

    score, acc = model.evaluate(x_test, y_test, verbose=0)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# In[15]:

import json
import random

def data():
    # Set the seed to achieve same sorting
    random.seed(6781693213)

    ids = list(range(0, 99))
    random.shuffle(ids)

    ids_train = ids[:80]
    ids_test = ids [80:]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    with open("modified_data.txt") as data:
        rows = json.load(data)

        x_train = np.take(rows[0], ids_train).reshape(-1, 1, 1)
        x_test = np.take(rows[0], ids_test).reshape(-1, 1, 1)
        y_train = (np.take(rows[2], ids_train) - 1).reshape(-1, 1)
        y_test = (np.take(rows[2], ids_test) - 1).reshape(-1, 1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test

# In[16]:


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
print(best_run)
