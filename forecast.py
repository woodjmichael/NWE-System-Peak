"""###############################################################################################
#                                                                                                #
#   forecast.py                                                                                  #
#                                                                                                #
#   keras      2.3.1                                                                             #
#   numpy      1.19.2                                                                            #
#   pandas     1.1.3                                                                             #
#   python     3.7.9                                                                             #
#   tensorflow 2.0.0                                                                             #
#                                                                                                #
###############################################################################################"""

import os, sys, platform
from numpy.core.numeric import NaN
import pandas as pd
import numpy as np
from numpy import isnan
from scipy.stats import moment, skew, kurtosis
#from sklearn.metrics import mean_squared_error as sk_mse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import datetime as dt
from pprint import pprint
from copy import deepcopy
#import seaborn as sns
#import emd


"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           functions                                            #
#                                                                                                #
#                                                                                                #
###############################################################################################"""


def config(plot_theme,seed):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert sys.version_info >= (3, 5)

    if plot_theme=='dark':
        plt.style.use('dark_background')
    elif plot_theme=='light':                
        plt.rcParams['axes.prop_cycle']
        #sns.set_style("whitegrid")
    pd.set_option('precision', 2)

    np.random.seed(seed)
    tf.random.set_seed(seed)

    if platform.system() == 'Darwin':   # macos
        IS_COLAB = False
    else: # probably colab
        IS_COLAB = True

        from google.colab import files
        from google.colab import drive
        drive.mount('/content/drive')
        
        # gpu_info = !nvidia-smi
        # gpu_info = '\n'.join(gpu_info)
        # if gpu_info.find('failed') >= 0:
        #   print('No GPU found')
        # else:
        #   print(gpu_info)

        # from psutil import virtual_memory
        # ram_gb = virtual_memory().total / 1e9
        # print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

    return IS_COLAB        

def get_data(site,IS_COLAB,additional_features,onehot_peak=False):
    features = deepcopy(additional_features)

    if IS_COLAB:
        path = '/content/drive/MyDrive/Data/'
    else:
        path = '~/Google Drive/My Drive/Data/'

    if site == 'lajolla':
        features.append('Load (kW)')
        df = import_data_lajolla(path)[features]
    elif site == 'lajolla_zerovals':
        features.append('Load (kW)')
        df = import_data_lajolla_zerovals(path)[features]
    elif site == 'northside':
        features.append('Load (kW)')
        df = import_data_northside(path)[features]
    elif site == 'hyatt':
        features.append('Load (kW)')
        df = import_data_hyatt(path)[features]
    elif site == 'nwe':     
        df = import_data_nwe(path, 'actual')
        df['T'] = import_temp_nwe(path)['T']


    if onehot_peak: # make one-hot vector of the peak time period
        df['peak'] = np.zeros(df.shape[0],dtype=int)

        # one-hot vector denoting peaks 
        df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:,0], 'peak'] = 1   

        df = df[['peak']]
             

    return df # numpy data vector                     

def import_data_nwe(path,feature):
    # feature = actual | forecast

    # import raw data from OATI OASIS
    filename = path + 'NWE/ca_'+feature+'.csv'
    df = pd.read_csv( filename, comment='#', parse_dates=['Date'], index_col=['Date'])                        

    # convert matrix of local-time values to vector of standard time values
    vec = []
    for __, row in df.iterrows():
        if isnan(row['2nd HR 2']): 
            for h in range(1,25): 
                if isnan(row['Hr %d'%h]): pass
                else: vec.append(row['Hr %d'%h])        
        else:
            for h in range(1,3):
                if isnan(row['Hr %d'%h]): pass
                else: vec.append(row['Hr %d'%h])
            vec.append(row['2nd HR 2'])
            for h in range(3,25):
                if isnan(row['Hr %d'%h]): pass
                else: vec.append(row['Hr %d'%h])
            
    nv =  np.asarray(vec)  # list to np array

    for val in nv: 
        if np.isnan(val): print('nan!') # check again for nan's

    # stuff into data frame (could be nice to have datetiem index)
    dates = pd.date_range( start=df.index.min(),
                            periods=len(vec),
                            freq='H')
    df = pd.DataFrame(nv,index=dates,columns=[feature])                            

    return df.loc['2007-7-9':'2021-4-27'] # these dates for consistency

def import_temp_nwe(path):
    filename = path + 'NWE/temp_mt-helena.csv'
    df = pd.read_csv(   filename,   
                        comment='#',                 
                        parse_dates=['Date'],
                        index_col=['Date'])                            
    df['T'] = pd.to_numeric(df['HourlyDryBulbTemperature degF'], errors='coerce')
    df = df[['T']]
    df = df.fillna(method='ffill')
    df = df.sort_index()
    df = df.resample('60T').mean()
    df = df.fillna(method='ffill')
    df = df['2015':]

    print('')
    print('length',len(df))
    print('dtype',df['T'].dtype)
    print('begin', df.index[0])
    print('end',df.index[-1])
    print('max dt',df.index.to_series().diff().max())
    print('min dt',df.index.to_series().diff().min())
    print('')
    return df                        


def import_data_lajolla(path):
    filename = path + 'lajolla_load_IMFs.csv'

    df = pd.read_csv(   filename,   
                        comment='#',                 
                        parse_dates=['Datetime (UTC-8)'],
                        index_col=['Datetime (UTC-8)'])

    return df

def import_data_lajolla_zerovals(path):
    filename = path + 'lajolla_load_IMFs_zerovals.csv'

    df = pd.read_csv(   filename,   
                        comment='#',                 
                        parse_dates=['Datetime (UTC-8)'],
                        index_col=['Datetime (UTC-8)'])

    return df    

def import_data_hyatt(path):
    filename = path + 'hyatt_load_IMFs.csv'

    df = pd.read_csv(   filename,   
                        comment='#',                 
                        parse_dates=['Datetime (UTC-10)'],
                        index_col=['Datetime (UTC-10)']) 

    return df.loc[:'2019-10-5'] # integer number of days 

def import_data_northside(path):
    filename = path + 'northside_load.csv'

    df = pd.read_csv(   filename,   
                        comment='#',                 
                        parse_dates=['Datetime MT'],
                        index_col=['Datetime MT']) 

    return df.loc[:'2021-04-29'] # integer number of days

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

def batchify_single_series(sv,b,no): #no = input + output size
    s = sv.reshape(b,no)
    return s[..., np.newaxis].astype(np.float32)

def batchify_single_series_sliding_window(sv,n,o): # n = input size, o = output size
    df = pd.DataFrame(sv, columns=['t'])
    for i in range(1,n+o):
        df['t+%d'%i] = df['t'].shift(-i)
    df = df.dropna()
    d = df.values
    return d[...,np.newaxis]

def convert_to_onehot(y):
    y2 = np.zeros((y.shape[0],y.shape[1]),dtype=int)
    for i in range(y.shape[0]):
        y2[i,np.argmax(y[i,:])] = 1
    return y2    

def mean_squared_error(y,y_hat):
    e = y - y_hat
    e2 = e * e
    return np.mean(e2)

def root_mean_squared_error(y,y_hat):
    mse = mean_squared_error(y,y_hat)
    return np.sqrt(mse)    

def accuracy_of_onehot(y_true,y_pred):
    # matrices must be one hots
    n_correct = []
    n_batches = y_true.shape[0]
    for i in range(n_batches):
        are_equal = np.array_equal(y_true[i,:], y_pred[i,:])
        n_correct.append(are_equal) 
    return sum(n_correct)/n_batches # return a value 0 to 1

def accuracy_of_onehot_vector(y_true,y_pred):
    # vectors must be one hots
    n_batches = y_true.shape[0]
    return np.sum(y_true*y_pred) / n_batches

def accuracy_of_NON_onehot_wrt_peaks(y_true,y_pred):
    # matrix must NOT be one hots
    # returns a value 0 to 1
    n_correct = []
    n_batches = y_true.shape[0]
    for i in range(n_batches):
        are_equal = np.argmax(y_true[i,:]) == np.argmax(y_pred[i,:])
        n_correct.append(are_equal) #
    return sum(n_correct)/n_batches # return a value 0 to 1

def nwe_forecast_accuracy(y_true, y_pred):
    y_pred = convert_to_onehot(y_pred)
    return accuracy_of_onehot(y_true, y_pred)

def daily_accuracy_on_rmse_basis(y_true,y_pred1,y_pred2):
    # matrices must NOT be one hots
    n_correct = []
    n_batches = y_true.shape[0]
    for i in range(n_batches):
        rmse1 = root_mean_squared_error(y_true[i,:],y_pred1[i,:])
        rmse2 = root_mean_squared_error(y_true[i,:],y_pred2[i,:])
        is_better = rmse1 > rmse2
        n_correct.append(is_better)    
    return sum(n_correct)/n_batches # return a value 0 to 1    

def naive_persistence(X,y,o,l,Lmax): # l = lag, o = output dim
    n = X.shape[1]
    begin = n - l
    end = begin + o
    y_pred = X[:,begin:end,0]
    rmse = np.mean(root_mean_squared_error(y,y_pred)) * Lmax
    acc  = accuracy_of_onehot(y,y_pred)
    return y_pred, rmse, acc        

def linear_regression(e,X_train,y_train,X_valid,y_valid,n,h,o,l,Lmax):
    t0 = dt.datetime.now()
    __, rmse_np, __ = naive_persistence(X_valid, y_valid, o, l, Lmax)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[n-h,1]),
        keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)

    rmse = np.mean(root_mean_squared_error(y_valid,y_pred))*Lmax
    skill = (rmse_np - rmse)/rmse_np
    ret = {'skill_np':skill}

    t = dt.datetime.now() - t0
    ret['minutes'] = t.total_seconds()/60
    return ret, y_pred, hx   

def deep_rnn(e,X_train,y_train,X_valid,y_valid,n,h,o,u,l,fcast,Lmax):
    t0=dt.datetime.now()

    __, rmse_np, acc_np = naive_persistence(X_valid,y_valid,o,l,Lmax)

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(units=u,return_sequences=True,input_shape=[None,X_valid.shape[2]]),
        keras.layers.SimpleRNN(units=u),
        keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)    
    print('rnn u{} e{} eval {:.6f}'.format(u,e,model.evaluate(X_valid, y_valid, verbose=0)))

    if fcast!='peak':
        rmse = np.mean(root_mean_squared_error(y_valid,y_pred))*Lmax 
        skill = (rmse_np - rmse)/rmse_np
        ret = {'skill_np':skill}
    if fcast=='peak':
        y_pred = convert_to_onehot(y_pred)
        acc = accuracy_of_onehot(y_valid,y_pred)
        dacc = acc - acc_np
        ret = {'diff acc':dacc}

    t = dt.datetime.now() - t0
    ret['minutes'] = t.total_seconds()/60
    return ret, y_pred, hx    

def lstm(e,X_train,y_train,X_valid,y_valid,n,h,o,u,l,fcast,Lmax):
    t0=dt.datetime.now()
    __, rmse_np, acc_np = naive_persistence(X_valid, y_valid, o, l, Lmax)

    model = keras.models.Sequential([
        keras.layers.LSTM(units=u,return_sequences=True,input_shape=[None,X_valid.shape[2]]),
        keras.layers.LSTM(units=u),
        keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)

    if fcast!='peak':
        rmse = np.mean(root_mean_squared_error(y_valid,y_pred))*Lmax 
        skill = (rmse_np - rmse)/rmse_np
        ret = {'skill_np':skill}
    if fcast=='peak':
        y_pred = convert_to_onehot(y_pred)
        acc = accuracy_of_onehot(y_valid,y_pred)
        dacc = acc - acc_np
        ret = {'diff acc':dacc}

    t = dt.datetime.now() - t0
    ret['minutes'] = t.total_seconds()/60
    
    print('lstm u{} e{}'.format(u,e),ret)
    return ret, y_pred, hx

def lstm_s2s(e,X_train,y_train,Y_train,X_valid,y_valid,Y_valid,n,h,o,u,l, fcast, Lmax):
    if fcast=='peak':
        print('\n\nerror: lstm_s2s() not updated for peak forecast\n\n')
        quit()

    t0=dt.datetime.now()
    __, rmse_np, acc_np = naive_persistence(X_valid, y_valid, o, l, Lmax)

    model = keras.models.Sequential([
        keras.layers.LSTM(u, return_sequences=True, input_shape=[None, X_valid.shape[2]]),
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


def nwe_inhouse_forecast(X_valid,y_valid,b,n,o,l,s1,s2,Lmax,fcast,IS_COLAB):
    __, rmse_np, acc_np = naive_persistence(X_valid, y_valid, o, l, Lmax)

    if IS_COLAB:
      path = '/content/drive/MyDrive/Data/'
    else:
      path = '../../../Google Drive/Data/'
      

    vF = import_data_nwe(path, 'forecast').values.flatten()
    if fcast == 'peak':
        df = import_data_nwe(path, 'forecast')
        df['peak'] = np.zeros(df.shape[0],dtype=int)
        df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:,0], 'peak'] = 1   
        df = df[['peak']]
        vF = df.values.flatten()

    mF = batchify_single_series(vF,b,n+o)/Lmax # matrix
    y_pred = mF[s1:s2, -o:, 0] # (to compare accuracy on same period)


    if fcast!='peak':
        rmse = np.mean(root_mean_squared_error(y_valid,y_pred))*Lmax 
        skill = (rmse_np - rmse)/rmse_np
        ret = {'skill_np':skill}
    if fcast=='peak':
        y_pred = convert_to_onehot(y_pred)
        acc = accuracy_of_onehot(y_valid,y_pred)
        dacc = acc - acc_np
        ret = {'diff acc':dacc}

    return ret, y_pred        

def plot_predictions(X,y,y_valid_data,y_pred,n,h,o,fcast,Lmax,batch,title):
    t1, t2 = np.arange(0,n), np.arange(n,n+o) # t1 inputs, t2 outputs    
    plt.figure(num=None, figsize=(10, 7), dpi=80)

    if o == 1: # single point forecast, likely sliding
        t = np.arange(24) # 24 points per day
        for model in y_pred:
            begin = batch * 24 # 24 points per day
            end   = (batch+1) * 24 # 24 points per day 
            rmse_batch = root_mean_squared_error(
                                                    y[begin:end],
                                                    y_pred[model][begin:end] )*Lmax
            plt.plot(   t,y[begin:end] * Lmax,
                        t,y_pred[model][begin:end] * Lmax,
                        label='y predict {} (rmse {:.2f})'.format(model,rmse_batch)) 

    elif fcast=='peak' and n==24 and o==24:
        plt.plot(t1, X[batch,   :,1]*Lmax,           label='X')

        plt.plot(t2, X[batch,   :,1]*Lmax,           label='y predict naive persistence')
        plt.plot(t2, y_valid_data[batch, :]*Lmax,    label='y true')

        y_peak = np.argmax(y[batch,:])
        title += ': true peak = h{}'.format(y_peak)
        for model in y_pred:
            predh = np.argmax(y_pred[model][batch,:]*Lmax)
            title += ', {} = h{}'.format(model,predh)                        

    elif fcast=='':
        plt.plot(t1, X[batch,:,0]*Lmax,              label='X')
        plt.plot(t2, y[batch,:]*Lmax,                label='y true') 

        for model in y_pred:
            rmse_batch = root_mean_squared_error(y[batch,:],y_pred[model][batch,:])*Lmax
            plt.plot(t2, y_pred[model][batch,:]*Lmax,   label='y predict {} (rmse {:.2f})'.format(model,rmse_batch))          
            
        plt.title(title + ': batch ' + str(batch) + ' inputs \'X\', targets and outputs \'y\'') 
        


        plt.title(title)

    


    plt.ylabel('Load (kW)')
    plt.xlabel('Timestep')
    plt.legend()
    plt.show()    
   

def plot_training(hx,first_epoch):
    for model in hx:
        if model == 'reg':
            pass
        else:
            plt.figure(num=None, figsize=(10, 7), dpi=80)
            plt.plot(hx[model].history['loss'][first_epoch:]) 
            #plt.plot(hx.history['val_loss'])
            plt.title('{} training'.format(model))
            plt.ylabel('model loss (nMSE)')
            plt.xlabel('epoch after {}'.format(first_epoch))
            #plt.legend(['Training', 'Validation'], loc='upper right')
            plt.show() 

def plot_every_day(df,day_of_week=-1,dppd=24,alpha=0.04):
    if day_of_week >= 0:
        df = df[df.index.weekday == day_of_week]

    num_days = int(df.shape[0]/dppd)
    d = df.values.reshape(num_days,dppd).T
    t = np.arange(0,24,24/dppd,dtype=float)

    plt.figure(num=None, figsize=(10,7),dpi=80)
    plt.plot(t,d,alpha=alpha)
    plt.xlim([0,24])
    plt.ylim([0,2000])
    plt.xlabel('Hour of Day')
    plt.ylabel('Load (kW)')
    plt.xticks(np.arange(0, 25, 1))
    plt.show()             

def print_inputs(X_train,y_train,b,n,h,o,u,e,fcast,site,feats):
    print('')
    print('Site:',site)
    print('Forecast Type:',fcast)
    print('Additional Features:',feats)
    print('Batches:',b)
    print('Input timesteps:',n)
    print('Forecast horizon timesteps (h+1):',h+1)
    print('Output timesteps:',o)
    print('Units (RNN/LSTM):',u)
    print('Epochs:',e)
    print('X_train shape:',X_train.shape)
    print('y_train shape:',y_train.shape)
    print('')        

def print_results(res,X_valid,y_valid,o,l,Lmax,fcast,np_only=False):
    print('')
    __, rmse_np, acc_np = naive_persistence(X_valid,y_valid,o,l,Lmax)

    # show np
    if fcast=='peak': print('np forecast accuracy {:.3f}'.format(acc_np))  
    else: print('np forecast rmse (kW) {:.1f}'.format(rmse_np))
    print('')
    #print(pd.DataFrame(data=res).drop(['runtime']).T.to_string())
    if not np_only: print(pd.DataFrame(data=res).T.to_string())
    print('')