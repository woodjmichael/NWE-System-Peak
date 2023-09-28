# sandbox.py

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
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

def linear_regression(X_train,y_train,X_valid,y_valid,n,h,o):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[n-h,1]),
        keras.layers.Dense(o),
    ])
    #opt = keras.optimizer.Adam(lr=0.01)
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,
                    validation_data=(X_valid,y_valid),
                    epochs=20,verbose=0)
    y_pred = model.predict(X_valid)
    mse = np.mean(mean_squared_error(y_valid,y_pred)) 
    print('reg eval',model.evaluate(X_valid, y_valid))

    plot_learning_curves(hx.history['loss'], hx.history['val_loss'])
    plt.plot()

    return y_pred, mse

def deep_rnn(X_train,y_train,X_valid,y_valid,n,h,o,u,e):
    model = keras.models.Sequential([
        keras.layers.SimpleRNN(units=u,return_sequences=True,input_shape=[None,1]),
        keras.layers.SimpleRNN(units=u),
        keras.layers.Dense(o),
    ])
    #opt = keras.optimizer.Adam(lr=0.01)
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=e,verbose=0)
    y_pred = model.predict(X_valid)
    mse = np.mean(mean_squared_error(y_valid,y_pred))
    print('rnn eval',model.evaluate(X_valid, y_valid))  

    plot_learning_curves(hx.history['loss'], hx.history['val_loss'])
    plt.show()

    return y_pred, mse  

def deep_rnn_s2s(X_train, y_train_s2s, X_valid, y_valid_s2s, n, h, o, u, e):
    rnn = keras.models.Sequential([
        keras.layers.SimpleRNN(units=u, return_sequences=True, input_shape=[None,1]),
        keras.layers.SimpleRNN(units=u, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(o)),
    ])
    #opt = keras.optimizer.Adam(lr=0.01)
    rnn.compile(loss='mse', optimizer='Adam', metrics=[last_time_step_mse])
    rnn.fit(X_train, y_train_s2s, epochs=e, verbose=0)
    y_pred_s2s = rnn.predict(X_valid)
    mse = np.mean(mean_squared_error(y_valid_s2s[:,-1], y_pred_s2s[:,-1]))  
    return y_pred_s2s, mse  

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:,-1], Y_pred[:,-1]) 

def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])

def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 0.05])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)    

#
# setup
#     
t_0 = dt.datetime.now()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

#
# data
#     

b = 10000 # batches
n = 50 # batch size
h = 0 # forecast horizon (timesteps skipped between X and y)
o = 10 # model output size
u = 20 # rnn units


s = generate_time_series(10000, n + o) # s[batches,timesteps]
#sv = generate_single_time_series(b, n + o) # s[batches,timesteps]
#s = batchify_single_series(sv, b, n + o)

X_train, y_train = s[:7000, :n-h],     s[:7000,     -o:, 0]
X_valid, y_valid = s[7000:9000, :n-h], s[7000:9000, -o:, 0]
X_test,  y_test  = s[9000:, :n-h],     s[9000:,     -o:, 0]

# sequence to sequence target preparation
Y = np.empty((10000, n, 10))
for step_ahead in range(1,10+1): # each target is a seris of 10D vectors
    Y[:, :, step_ahead - 1] = s[:, step_ahead : step_ahead + n, 0]

y_train_s2s = Y[:7000]
y_valid_s2s = Y[7000:9000]
y_test_s2s  = Y[9000:] 

print('')
print('Batch size',n)
print('Forecast horizon',h)
print('Model output size',o)
print('X_train shape',X_train.shape)
print('y_train shape',y_train.shape)
print('y_train_s2s shape',y_train_s2s.shape)
print('')

y_pred, mse = {}, {}

#
# models
#

y_pred['naive'], mse['naive'] = naive_persistence(X_valid, y_valid, o)

t_naive = dt.datetime.now()

y_pred['reg'], mse['reg'] = linear_regression(X_train, y_train, 
                                              X_valid, y_valid, 
                                              n, h, o)

t_reg = dt.datetime.now()

y_pred['rnn'], mse['rnn'] = deep_rnn(X_train, y_train, 
                                     X_valid, y_valid, 
                                     n, h, o, u, e=20)

t_rnn = dt.datetime.now()

#y_pred['rnn_s2s'], mse['rnn_s2s'] = deep_rnn_s2s(X_train, y_train_s2s, 
#                                                 X_valid, y_valid_s2s, 
#                                                 n, h, o, u, e=20)


#
# model: geron
#

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

t_1 = dt.datetime.now()

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01), metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid),
                    verbose=0)

Y_pred = model.predict(X_valid)
mse['geron'] = np.mean(last_time_step_mse(Y_valid, Y_pred))
print('geron eval',model.evaluate(X_valid, Y_valid))  

t_geron = dt.datetime.now()

#plot_learning_curves(history.history['loss'], history.history['val_loss'])
#plt.show()

#
# results
#

print('mse',mse,'\n')

t_end = dt.datetime.now()
print('runtime',t_end - t_0)
print('t_naive - t_0',t_naive - t_0)
print('t_reg - t_naive',t_reg - t_naive)
print('t_rnn - t_reg', t_rnn - t_reg)
print('t_geron - t_1',t_geron - t_1)



#
# plot on new data
#

# np.random.seed(43)

# series = generate_time_series(1, 50 + 10)
# X_new, Y_new = series[:, :50, :], series[:, 50:, :]
# Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]  

# plot_multiple_forecasts(X_new, Y_new, Y_pred)
# plt.show()

