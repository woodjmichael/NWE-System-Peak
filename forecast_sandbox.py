# sandbox.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def generate_time_series(b, n): # batch size, n_steps
    f1, f2, o1, o2 = np.random.rand(4, b, 1)
    t = np.linspace(0,1,n)
    s = 0.5 * np.sin((t - o1) * (f1 * 10 + 10)) # wave 1
    s += 0.2 * np.sin((t - o2) * (f2 * 20 + 20)) # + wave 2
    s += 0.1 * (np.random.rand(b, n) - 0.5) # + noise
    return s[..., np.newaxis].astype(np.float32)

def generate_single_time_series(b, n): # batch size, n_steps
    f1, f2, o1, o2 = np.random.rand(4, b, 1)
    t = np.linspace(0,1,n)
    s = 0.5 * np.sin((t - o1) * (f1 * 10 + 10)) # series = wave 1
    s += 0.2 * np.sin((t - o2) * (f2 * 20 + 20)) # + wave 2
    s += 0.1 * (np.random.rand(b, n) - 0.5) # + noise

    return s.reshape(b*n,1)[:,0] # 1d


def batchify_single_series(sv,b,n):
    s = sv.reshape(b,n)
    return s[..., np.newaxis].astype(np.float32)


# def batchify_time_series(b,n):
#     t = np.linspace(0,1,b*n)
#     s = 0.5 * np.sin((t - o1) * (f1 * 10 + 10)) # wave 1
#     s += 0.2 * np.sin((t - o2) * (f2 * 20 + 20)) # + wave 2
#     s += 0.1 * (np.random.rand(b, n) - 0.5) # + noise
#     return s[..., np.newaxis].astype(np.float32)

def mean_squared_error(y,y_hat):
    e = y - y_hat
    e2 = e * e
    return np.mean(e2)

def naive_persistence(X_valid,y_valid,o):
    y_pred = X_valid[:,-o:,0]
    mse = np.mean(mean_squared_error(y_valid,y_pred))

    return y_pred, mse

def linear_regression(X_train,y_train,n,h,o):
    reg = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[n-h,1]),
    keras.layers.Dense(o),
    ])
    reg.compile(loss='mse',optimizer='Adam')
    reg.fit(X_train,y_train,epochs=20,verbose=0)
    y_pred = reg.predict(X_valid)
    mse = np.mean(mean_squared_error(y_valid,y_pred))    
    return y_pred, mse

def deep_rnn(X_train,y_train,n,h,o):
    rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(20,return_sequences=True,input_shape=[None,1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(o),
    ])
    rnn.compile(loss='mse',optimizer='Adam')
    rnn.fit(X_train,y_train,epochs=20,verbose=0)
    y_pred = rnn.predict(X_valid)
    mse = np.mean(mean_squared_error(y_valid,y_pred))  
    return y_pred, mse  

#
# setup
#     
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# data
#     

b = 10000 # batches
n = 50 # batch size
h = 0 # forecast horizon (timesteps skipped between X and y)
o = 1 # model output size


#s = generate_time_series(10000, n + o) # s[batches,timesteps]
sv = generate_single_time_series(b, n + o) # s[batches,timesteps]
s = batchify_single_series(sv, b, n + o)

X_train, y_train = s[:7000, :n-h],     s[:7000,     -o:, 0]
X_valid, y_valid = s[7000:9000, :n-h], s[7000:9000, -o:, 0]
X_test,  y_test  = s[9000:, :n-h],     s[9000:,     -o:, 0]

print('')
print('Batch size',n)
print('Forecast horizon',h)
print('Model output size',o)
print('X_train shape',X_train.shape)
print('y_train shape',y_train.shape)
print('')

y_pred, mse = {}, {}

#
# model: naive persistance
#

y_pred['naive'], mse['naive'] = naive_persistence(X_valid, y_valid, o)


#
# model: linear regression
#

y_pred['reg'], mse['reg'] = linear_regression(X_valid, y_valid, n, h, o)

#
# model: deep rnn
#

y_pred['rnn'], mse['rnn'] = deep_rnn(X_valid, y_valid, n, h, o)

#
# results
#

print('mse',mse,'\n')