# forecast.py
# python 3.8.3
# pandas 1.0.5
# tensorflow 2.3.1
# keras 2.4.0

# 'base' conda env

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from _nwe import import_data
import datetime as dt
from pprint import pprint

#
# functions
#

def import_data_lajolla():
    filename = '/Users/mjw/Google Drive/Data/lajolla_load_processed.csv'

    fields = ['Datetime (UTC-8)','Load (kW)']

    df = pd.read_csv(   filename,   
                        comment='#',                 
                        parse_dates=['Datetime (UTC-8)'],
                        index_col=['Datetime (UTC-8)'],
                        usecols=fields)   

    return df

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

def mean_squared_error(y,y_hat):
    global dmax
    y = y * dmax
    y_hat = y_hat * dmax
    e = y - y_hat
    e2 = e * e
    return np.mean(e2)

def root_mean_squared_error(y,y_hat):
    mse = mean_squared_error(y,y_hat)
    return np.sqrt(mse)

def print_inputs(X_train,y_train,b,n,h,o,u):
    print('')
    print('Batches',b)
    print('Input dimension',n)
    print('Forecast horizon',h)
    print('Output dimension',o)
    print('Units',u)
    print('X_train shape',X_train.shape)
    print('y_train shape',y_train.shape)
    print('')     

def naive_persistence(X_valid,y_valid,o):
    y_pred = X_valid[:,-o:,0]
    rmse = np.mean(root_mean_squared_error(y_valid,y_pred))
    return y_pred, rmse    

def linear_regression(e,X_train,y_train,X_valid,y_valid,n,h,o):
    t0 = dt.datetime.now()
    __, rmse_np = naive_persistence(X_valid, y_valid, o)

    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[n-h,1]),
    keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)
    rmse = np.mean(root_mean_squared_error(y_valid,y_pred))
    skill = rmse_np - rmse

    t = dt.datetime.now() - t0
    ret = {'epochs':e,'units':0,'skill_np':skill,'runtime':t}
    return ret, y_pred, hx

def deep_rnn(e,X_train,y_train,X_valid,y_valid,n,h,o,u):
    t0=dt.datetime.now()
    __, rmse_np = naive_persistence(X_valid, y_valid, o)

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(units=u,return_sequences=True,input_shape=[None,1]),
        keras.layers.SimpleRNN(units=u),
        keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)
    
    print('rnn eval',model.evaluate(X_valid, y_valid)) 
    rmse = np.mean(root_mean_squared_error(y_valid,y_pred)) 
    skill = rmse_np - rmse

    t =  dt.datetime.now() - t0
    ret = {'epochs':e,'units':u,'skill_np':skill,'runtime':t}
    return ret, y_pred, hx

def lstm(e,X_train,y_train,X_valid,y_valid,n,h,o,u):
    t0=dt.datetime.now()
    __, rmse_np = naive_persistence(X_valid, y_valid, o)

    model = keras.models.Sequential([
        keras.layers.LSTM(units=u,return_sequences=True,input_shape=[None,1]),
        keras.layers.LSTM(units=u),
        keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)

    print('lstm eval',model.evaluate(X_valid, y_valid)) 
    rmse = np.mean(root_mean_squared_error(y_valid,y_pred)) 
    skill = rmse_np - rmse

    t=dt.datetime.now() - t0
    ret = {'epochs':e,'units':u,'skill_np':skill,'runtime':t}
    return ret, y_pred, hx

def lstm_s2s(e,X_train,y_train,Y_train,X_valid,y_valid,Y_valid,n,h,o,u):
    t0=dt.datetime.now()
    __, rmse_np = naive_persistence(X_valid, y_valid, o)

    model = keras.models.Sequential([
        keras.layers.LSTM(u, return_sequences=True, input_shape=[None, 1]),
        keras.layers.LSTM(u, return_sequences=True),
        keras.layers.TimeDistributed(keras.layers.Dense(o))
    ])

    model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
    hx = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))

    y_pred = model.predict(X_valid)
    rmse = np.mean(root_mean_squared_error(y_valid,y_pred)) 
    skill = 0#rmse_np - rmse

    t=dt.datetime.now() - t0
    ret = {'epochs':e,'units':u,'skill_np':skill,'runtime':t}
    return ret, y_pred, hx    

def plot_predictions(X_valid,y_valid,y_valid_pred,n,h,o,k):
    t1, t2 = np.arange(0,n), np.arange(n,n+o) 

    y_valid_pred['np'],  __ = naive_persistence(X_valid, y_valid, o)
    
    plt.plot(t1, X_valid[k,:,0],                label='X')
    plt.plot(t2, y_valid[k,:],                  label='y')
    plt.plot(t2, y_valid_pred['np'][k,:],       label='y^ naive persistence')
    plt.plot(t2, y_valid_pred['reg'][k,:],      label='y^ regression')
    plt.plot(t2, y_valid_pred['rnn'][k,:],      label='y^ rnn')
    plt.plot(t2, y_valid_pred['lstm'][k,:],     label='y^ lstm')

    plt.title('validation set')    
    plt.legend()
    plt.show()    

def plot_training(hx):
    #plt.figure(num=None, figsize=(10, 7), dpi=160)
    plt.plot(hx.history['loss']) 
    #plt.plot(hx.history['val_loss'])
    plt.title('Training')
    plt.ylabel('Model Loss (nMSE)')
    plt.xlabel('Epoch')
    #plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()   


################ main execution ################

#
# setup
#     

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#
# data
#

dv = import_data('actual').values[:,0] # data vector

#df = import_data_lajolla()

b = 2525 # batches
n = 24 # number of timesteps (input)
h = 0 # horizon of forecast
o = 24 # output dimension
u = 50 # rnn/lstm units
e = 5000

d_fullscale = batchify_single_series(dv,b,n+o)
dmax = np.max(d_fullscale)
d = d_fullscale / dmax

s1 = int(.8*b) # split 1 (valid)
s2 = int(.9*b) # split 2 (test)

X_train, y_train = d[:s1,   :n-h], d[:s1,   -o:, 0] # features, targets 
X_valid, y_valid = d[s1:s2, :n-h], d[s1:s2, -o:, 0]
X_test,  y_test  = d[s2:,   :n-h], d[s2:,   -o:, 0]

print_inputs(X_train,y_train,b,n,h,o,u)


#
# models
#

res, y_valid_pred, hx, = {}, {}, {}

res['reg'], y_valid_pred['reg'], hx['reg'] = linear_regression(1000, X_valid, y_valid, X_valid, y_valid, n, h, o)

res['rnn'], y_valid_pred['rnn'], hx['rnn'] = deep_rnn(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u)

res['lstm'],y_valid_pred['lstm'],hx['lstm'] = lstm(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u)

#res['lstm_s2s'],y_valid_pred['lstm_s2s'],hx['lstm_s2s'] = lstm_s2s(e, X_valid, y_valid, Y_valid, X_valid, y_valid, Y_valid, n, h, o, u)

# units = [100,200,300,400,500,600,700,800,900,1000] 
# for u in units:
#     name1 = 'rnn %d'%u
#     name2 = 'lstm %d'%u
#     res[name1], y_valid_pred[name1], hx[name1] = deep_rnn(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u)
#     res[name2],y_valid_pred[name2],hx[name2] = lstm(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u)

# epochs = [25,100,500,1000]
# for e in epochs:
#     name = 'lstm %d'%e
#     res[name], y_valid_pred[name], hx[name] = lstm(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u)
#     plot_training(hx[name])


#
# results
#

df = pd.DataFrame(data=res).drop(['runtime']).T
print(df.to_string())

#plot_predictions(X_valid,y_valid,y_valid_pred,n,h,o,1)

#plot_training(hx['lstm'])

#
# outro
#

sys.stdout.write('\a') # beep  