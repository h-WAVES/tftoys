#coding:utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import  learn

LOG_DIR = './ops_logs'
TIMESTEPS = 5
RNN_LAYERS = [{'steps': TIMESTEPS}, {'steps': TIMESTEPS, 'keep_prob': 0.5}]
DENSE_LAYERS = [2]
TRAINING_STEPS = 130000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

def mean_squared_error(errors):
    return np.mean(np.square(errors))

def x_sin(x):
    return x * np.sin(x)

def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)

def split_data(data, val_size=0.1, test_size=0.1):
    n_test = int(round(len(data) * (1 - val_size)))
    n_val = int(round(len(data.iloc[:n_test]) * (1 - val_size)))
    df_train, df_val, df_test = 

def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    df_train, df_val, df_test = split(data, val_size, test_size)


def generate_data(fct, x, time_steps, sperate=False):
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        x_train, y_train, x_test = pre

regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
                                      n_classes=0,
                                      verbose=1,
                                      steps=TRAINING_STEPS,
                                      optimizer='Adagrad',
                                      learning_rate=0.03,
                                      batch_size=BATCH_SIZE)
X, y = generate_data(np.sin, np.linspace(0, 100, 10000), TIMESTEPS, seperate=False)
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=1000)
regressor.fit(X['train'], y['train'], validation_monitor, logdir=LOG_DIR)
mse = mean_squared_error(regressor.predict(X['test']), y['test'])
print ("Error: {}".format(mse))
