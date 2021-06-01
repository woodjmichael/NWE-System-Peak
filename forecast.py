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
import pandas as pd
import numpy as np
from numpy import isnan
from scipy.stats import moment, skew, kurtosis
from sklearn.metrics import mean_squared_error as sk_mse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import datetime as dt
from pprint import pprint
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

def import_data(site,fcast,IS_COLAB,feature):
    if IS_COLAB:
        path = '/content/drive/MyDrive/Data/'
    else:
        path = '../../../Google Drive/Data/'

    if site == 'lajolla':
        d = import_data_lajolla(path)[feature].values
    elif site == 'hyatt':
        d = import_data_hyatt(path)[feature].loc[:'2019-10-5'].values # integer number of days
    elif (site == 'nwe'):     
        d = import_data_nwe('actual', fcast, path)['actual'].loc['2007-7-9':'2021-4-27'].values # align dates with forecast

    return d # numpy data vector               

def import_data_nwe(source, fcast, path):

    filename = path + 'NWE/ca_' + source + '.csv'

    df = pd.read_csv(   filename,   
                        comment='#',                 
                        parse_dates=['Date'],
                        index_col=['Date'])

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
            
    nv =  np.asarray(vec)

    for val in nv: 
        if np.isnan(val): print('nan!')

    dates = pd.date_range( start=df.index.min(),
                            periods=len(vec),
                            freq='H')

    df = pd.DataFrame(nv,index=dates,columns=[source])                            

    if fcast=='peak':
        df['peak'] = np.zeros(df.shape[0],dtype=int)

        # one-hot vector denoting peaks 
        df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:,0], 'peak'] = 1

    return df

def import_data_lajolla(path):
    filename = path + 'lajolla_load_IMFs.csv'

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
    global Lmax
    e = y - y_hat
    e2 = e * e
    return np.mean(e2)

def root_mean_squared_error(y,y_hat):
    mse = mean_squared_error(y,y_hat)
    return np.sqrt(mse)  

def naive_persistence(X,y,o,Lmax):
    y_pred = X[:,-o:,0]
    rmse = np.mean(root_mean_squared_error(y,y_pred)) * Lmax
    acc  = accuracy_of_onehot(y,y_pred)
    return y_pred, rmse, acc         

def convert_to_onehot(y):
    y2 = np.zeros((y.shape[0],y.shape[1]),dtype=int)
    for i in range(y.shape[0]):
        y2[i,np.argmax(y[i,:])] = 1
    return y2

def accuracy_of_onehot(y_true,y_pred):
    right = []
    batches = y_true.shape[0]
    for i in range(batches):
        right.append(np.array_equal(y_true[i,:], y_pred[i,:])) # 'True' if the rows are equal
    return sum(right)/batches

def nwe_forecast_accuracy(y_true, y_pred):
    y_pred = convert_to_onehot(y_pred)
    return accuracy_of_onehot(y_true, y_pred)

def linear_regression(e,X_train,y_train,X_valid,y_valid,n,h,o):
    t0 = dt.datetime.now()
    __, rmse_np, __ = naive_persistence(X_valid, y_valid, o)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[n-h,1]),
        keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)

    rmse = np.mean(root_mean_squared_error(y_valid,y_pred))*Lmax
    skill = (rmse_np - rmse)/rmse_np

    t = dt.datetime.now() - t0
    ret = {'skill_np':skill,'runtime':t}
    return ret, y_pred, hx    

def deep_rnn(e,X_train,y_train,X_valid,y_valid,n,h,o,u,fcast):
    t0=dt.datetime.now()

    __, rmse_np, acc_np = naive_persistence(X_valid, y_valid, o)

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
    if fcast=='peak':
        y_pred = convert_to_onehot(y_pred)

        #acc_np = accuracy_of_onehot(y_valid,y_np)
        acc = accuracy_of_onehot(y_valid,y_pred)

        skill = acc - acc_np

    t =  dt.datetime.now() - t0
    ret = {'skill_np':skill,'runtime':t}
    return ret, y_pred, hx    

def lstm(e,X_train,y_train,X_valid,y_valid,n,h,o,u,fcast,Lmax):
    t0=dt.datetime.now()
    __, rmse_np, acc_np = naive_persistence(X_valid, y_valid, o,Lmax)

    model = keras.models.Sequential([
        keras.layers.LSTM(units=u,return_sequences=True,input_shape=[None,X_valid.shape[2]]),
        keras.layers.LSTM(units=u),
        keras.layers.Dense(o),
    ])
    model.compile(loss='mse',optimizer='Adam')
    hx = model.fit(X_train,y_train,epochs=e,verbose=0)
    y_pred = model.predict(X_valid)
    print('lstm u{} e{} eval {:.6f}'.format(u,e,model.evaluate(X_valid, y_valid, verbose=0)))

    if fcast!='peak':
        rmse = np.mean(root_mean_squared_error(y_valid,y_pred))*Lmax 
        skill = (rmse_np - rmse)/rmse_np
    if fcast=='peak':
        y_pred = convert_to_onehot(y_pred)

        #acc_np = accuracy_of_onehot(y_valid,y_np)
        acc = accuracy_of_onehot(y_valid,y_pred)

        skill = acc - acc_np

    t=dt.datetime.now() - t0
    ret = {'skill_np':skill,'runtime':t}
    return ret, y_pred, hx

def lstm_s2s(e,X_train,y_train,Y_train,X_valid,y_valid,Y_valid,n,h,o,u,fcast):
    if fcast=='peak':
        print('\n\nerror: lstm_s2s() not updated for peak forecast\n\n')
        quit()

    t0=dt.datetime.now()
    __, rmse_np, acc_np = naive_persistence(X_valid, y_valid, o)

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

def plot_predictions(X,y,y_pred,n,h,o,fcast,Lmax,batch,title):
    t1, t2 = np.arange(0,n), np.arange(n,n+o) # t1 inputs, t2 outputs
    y_pred['np'],  __, __ = naive_persistence(X, y, o,Lmax)
    plt.figure(num=None, figsize=(10, 7), dpi=80)

    if not fcast=='peak':
        plt.plot(t1, X[batch,:,0],              label='X')
        plt.plot(t2, y[batch,:],                label='y true') 

        for model in y_pred:
            rmse_batch = root_mean_squared_error(y[batch,:],y_pred[model][batch,:])*Lmax
            plt.plot(t2, y_pred[model][batch,:],   label='y predict {} (rmse {:.2f} kW)'.format(model,rmse_batch))          
            
        plt.title(title + ': batch ' + str(batch) + ' inputs \'X\', targets and outputs \'y\'') 
        
    if fcast=='peak' and n==24 and o==24:
        plt.plot(t1, X[batch,   :,1],           label='X')

        plt.plot(t2, X[batch,   :,1],           label='y predict naive persistence')
        plt.plot(t2, X[batch+1, :,1],           label='y true')

        peakh = np.argmax(y[batch,:])
        forecasth_rnn = np.argmax(y_pred['rnn'][batch,:])
        forecasth_lstm = np.argmax(y_pred['lstm'][batch,:])

        plt.title(title + ' true peak=h{}, RNN forecast=h{}, LSTM forecast=h{}'.format(peakh,forecasth_rnn,forecasth_lstm))

    plt.ylabel('Load (scaled to max=1)')
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

def print_inputs(X_train,y_train,b,n,h,o,u,e,fcast,site,feats):
    print('')
    print('Site',site)
    print('Forecast type',fcast)
    print('Additional Features',feats)
    print('Batches',b)
    print('Input timesteps',n)
    print('Forecast horizon timesteps',h)
    print('Output timesteps',o)
    print('Units (RNN/LSTM)',units)
    print('Epochs',epochs)
    print('X_train shape',X_train.shape)
    print('y_train shape',y_train.shape)
    print('')        

def print_results(res,X_valid,y_valid,y_valid_nwef,o,Lmax,fcast):
    print('')
    __, rmse_np, acc_np = naive_persistence(X_valid,y_valid,o,Lmax)

    # show np
    if fcast=='peak': print('np forecast accuracy {:.3f}'.format(acc_np))  
    else: print('np forecast rmse (kW) {:.1f}'.format(rmse_np))
    print('')
    #print(pd.DataFrame(data=res).drop(['runtime']).T.to_string())
    print(pd.DataFrame(data=res).T.to_string())
    print('')

# def emd_sift_and_plot(df,data_col):
#     #imf = emd.sift.sift(df['load'].values)
#     imf = emd.sift.sift(df[data_col].values)

#     for i in range(imf.shape[1]):
#         #df_emd.insert(i+2,'IMF%s'%(i+1), imf[:,i])
#         df['IMF%s'%(i+1)] = imf[:,i]    

#     return df

"""###############################################################################################
##################################################################################################
##################################################################################################
####################################### main execution ###########################################
##################################################################################################
##################################################################################################
###############################################################################################"""


"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           config                                               #
#                                                                                                #
#                                                                                                #
###############################################################################################"""
IS_COLAB = config(plot_theme='light',seed=42) 

site = 'lajolla'     
fcast = 'normal' # normal, peak
feats = 'IMF3' # '', or 'IMFx'
b = 377 # batches (nwe = , lajolla 2 day = 377, hyatt 2 day = 528)
n = 96 # number of timesteps (input)
h = 0 # horizon of forecast
o = 96 # output timesteps
units = [25] 
epochs = [1000]
s1, s2 = int(.63*b), int(.9*b) # split 1 (train-valid), 2 (valid-test)



"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           data                                                 #
#                                                                                                #
#                                                                                                #
###############################################################################################"""

# load
vL = import_data(site,fcast,IS_COLAB,feature='Load (kW)') # load vector
mL = batchify_single_series(vL,b,n+o) # load matrix
Lmax = np.max(mL) 
mL = mL / Lmax # scale by max
d = mL # shape: (batches,timesteps)

if feats != '':
    vIMF3 = import_data(site,fcast,IS_COLAB,feature=feats) # load vector
    mIMF3 = batchify_single_series(vIMF3,b,n+o) # peaks matrix
    mIMF3 = mIMF3/np.max(mIMF3) # scale by max
    d = np.concatenate((mL,mIMF3),axis=2) # shape: (batches,timesteps,features)

# peaks
if fcast=='peak':
    vP = import_data_nwe('actual', fcast, IS_COLAB)['peak'    ].loc['2007-7-9':'2021-4-27'].values # peak one-hot vector
    mP = batchify_single_series(vP,b,n+o) # peaks matrix
    d = np.concatenate((mP,mL),axis=2) # shape: (batches,timesteps,features)

# X and y, split train-test-valid
X_train, y_train = d[:s1,   :n-h, :], d[:s1,   -o:, 0] # (features, targets)
X_valid, y_valid = d[s1:s2, :n-h, :], d[s1:s2, -o:, 0]
X_test,  y_test  = d[s2:,   :n-h, :], d[s2:,   -o:, 0]

y_pred, rmse, acc = naive_persistence(X_valid, y_valid, o, Lmax)

# nwe forecast
# vF = import_data_nwe('forecast', fcast, IS_COLAB)['forecast'].loc['2007-7-9':'2021-4-27'].values # vector
# mF = batchify_single_series(vF,b,n+o) # matrix
# X_valid_nwef, y_valid_nwef = mF[s1:s2, :n-h, :], mF[s1:s2, -o:, 0] # (to compare accuracy on same period)

print_inputs(X_train,y_train,b,n,h,o,units,epochs,fcast,site,feats)

"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           models                                               #
#                                                                                                #
#                                                                                                #
###############################################################################################"""

res, y_valid_pred, hx, = {}, {}, {}

# linear regression
#if fcast == 'normal': # can't use linear_regression() w/ multiple features (peaks, emd)
#    res['reg'], y_valid_pred['reg'], hx['reg'] = linear_regression(1000, X_valid, y_valid, X_valid, y_valid, n, h, o)

# simple rnn
#res['rnn'], y_valid_pred['rnn'], hx['rnn']  = deep_rnn(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u, fcast)

# lstm
#res['lstm'],y_valid_pred['lstm'],hx['lstm'] =     lstm(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u, fcast)

# lstm s2s
#res['lstm_s2s'],y_valid_pred['lstm_s2s'],hx['lstm_s2s'] = lstm_s2s(e, X_valid, y_valid, Y_valid, X_valid, y_valid, Y_valid, n, h, o, u,  fcast)

for u in units:
    for e in epochs:
        #name = 'rnn u{} e{}'.format(u,e)        
        #res[name], y_valid_pred[name], hx[name] = deep_rnn(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u, fcast)

        name = 'lstm u{} e{}'.format(u,e)
        res[name],y_valid_pred[name],hx[name] = lstm(e, X_valid, y_valid, X_valid, y_valid, n, h, o, u, fcast,Lmax)



"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           results                                              #
#                                                                                                #
#                                                                                                #
###############################################################################################"""

print_results(res,X_valid,y_valid,y_valid,o,Lmax,fcast)

plot_predictions(X_valid,y_valid,y_valid_pred,n,h,o,fcast,Lmax,batch=0,title='validation set')

#plot_predictions(X_test,y_test,y_valid_pred,n,h,o,fcast,batch=0,title='test set')

plot_training(hx, first_epoch=25)    

# a = X_train[0,:96,0]*Lmax
# b = y_train[0,:]*Lmax
# e2 = (a-b)*(a-b)
# rmse = np.sqrt(np.mean(e2))
# print('mean(a)',np.mean(a))
# print('mean(b)',np.mean(b))
# print('a[0]',a[0])
# print('b[0]',b[0])
# print('rmse',root_mean_squared_error(a,b))