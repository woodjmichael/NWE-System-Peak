# forecast.py
# python 3.8.3
# pandas 1.0.5
# tensorflow 2.3.1
# keras 2.4.0

# 'base' conda env

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from _nwe import import_data


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


def batchify_single_series(sv,b,no):
    s = sv.reshape(b,no)
    return s[..., np.newaxis].astype(np.float32)


# def batchify_time_series(b,n):
#     t = np.linspace(0,1,b*n)
#     s = 0.5 * np.sin((t - o1) * (f1 * 10 + 10)) # wave 1
#     s += 0.2 * np.sin((t - o2) * (f2 * 20 + 20)) # + wave 2
#     s += 0.1 * (np.random.rand(b, n) - 0.5) # + noise
#     return s[..., np.newaxis].astype(np.float32)

def mean_squared_error(y,y_hat):
    global dmax
    y = y * dmax
    y_hat = y_hat * dmax
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
    reg.fit(X_train,y_train,epochs=2000,verbose=0)
    y_pred = reg.predict(X_valid)
    mse = np.mean(mean_squared_error(y_valid,y_pred))    
    return y_pred, mse

def deep_rnn(X_train,y_train,n,h,o):
    rnn = keras.models.Sequential([
    keras.layers.SimpleRNN(100,return_sequences=True,input_shape=[None,1]),
    keras.layers.SimpleRNN(100),
    keras.layers.Dense(o),
    ])
    rnn.compile(loss='mse',optimizer='Adam')
    rnn.fit(X_train,y_train,epochs=2000,verbose=0)
    y_pred = rnn.predict(X_valid)
    mse = np.mean(mean_squared_error(y_valid,y_pred))  
    return y_pred, mse

def plot_all(X_valid,y_valid,y_valid_pred,n,h,o,k):
    t1 = np.arange(0,n)
    t2 = np.arange(n,n+o)
    
    plt.plot(t1, X_valid[k,:,0],            label='X')

    plt.plot(t2, y_valid[k,:],              label='y')
    plt.plot(t2, y_valid_pred['np'][k,:],   label='y^ naive persistence')
    plt.plot(t2, y_valid_pred['reg'][k,:],  label='y^ regression')
    plt.plot(t2, y_valid_pred['rnn'][k,:],  label='y^ rnn')

    plt.title('validation set')    
    plt.legend()
    plt.show()    

def print_inputs(X_train,y_train,b,n,h,o):
    print('')
    print('Batches',b)
    print('Input dimension',n)
    print('Forecast horizon',h)
    print('Output dimension',o)
    print('X_train shape',X_train.shape)
    print('y_train shape',y_train.shape)
    print('')    


#
# setup
#     

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# data
#

df = import_data('actual')
dv = df.values[:,0]

b = 2020 # number batches
n = 36 # input dimension
h = 0 # forecast horizon
o = 24 # output dimension

d_fullscale = batchify_single_series(dv,b,n+o)

dmax = np.max(d_fullscale)
d = d_fullscale / dmax

X_train, y_train = d[:1500,     :n-h], d[:1500,     -o:, 0]
X_valid, y_valid = d[1500:1750, :n-h], d[1500:1750, -o:, 0]
X_test,  y_test  = d[1750:,     :n-h], d[1750:,     -o:, 0]

print_inputs(X_train,y_train,b,n,h,o)

y_valid_pred, mse = {}, {}

#
# model: naive persistance
#

y_valid_pred['np'], mse['np'] = naive_persistence(X_valid, y_valid, o)


#
# model: linear regression
#

y_valid_pred['reg'], mse['reg'] = linear_regression(X_valid, y_valid, n, h, o)

#
# model: deep rnn
#

y_valid_pred['rnn'], mse['rnn'] = deep_rnn(X_valid, y_valid, n, h, o)

#
# results
#

print('mse',mse,'\n')

plot_all(X_valid,y_valid,y_valid_pred,n,h,o,0)
plot_all(X_valid,y_valid,y_valid_pred,n,h,o,1)
plot_all(X_valid,y_valid,y_valid_pred,n,h,o,2)
plot_all(X_valid,y_valid,y_valid_pred,n,h,o,3)
plot_all(X_valid,y_valid,y_valid_pred,n,h,o,4)
