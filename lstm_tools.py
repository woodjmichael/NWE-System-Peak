__version__ = 1.3

import os, sys, shutil

import yaml

from typing import Generator
from datetime import datetime

from random import shuffle

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load
#from statsmodels.tsa.stattools import adfuller # upgrade statsmodels (at least 0.11.1)
#from scipy.stats import shapiro, mode

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Embedding, Dropout
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, \
                                                                                ReduceLROnPlateau
from tensorflow.keras.backend import square, mean

import emd

print(tf.config.list_physical_devices('GPU'))

pd.options.display.float_format = '{:.2f}'.format

class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def read_yaml(config_file:str=None) -> dotdict:
    with open(config_file, 'r') as stream:
        d=yaml.safe_load(stream)
    cfg = dotdict(d)
    return cfg

def model_builder_kt(hp):
    n_features_x=1
    n_in=96
    n_out=96 
    path_checkpoint=f'lstm.keras'
    dropout=[0,0]
    loss='mse'
    batch_size=256

    model = Sequential()

    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    
    model.add( LSTM(hp_units,
                    return_sequences=True,
                    input_shape=(None, n_features_x,),
                    activation='relu') )
    
    # if dropout is not None:
    #     if dropout[0] != 0:
    #         model.add(Dropout(dropout[0]))
    
    # model.add( LSTM( hp_units,
    #                 return_sequences=True,
    #                 activation='relu') ) 
    
    if dropout is not None:
        if dropout[1] != 0:
            model.add(Dropout(dropout[1]))
        
    model.add( Dense( n_out, activation='sigmoid') )    

    if loss == 'custom':
        model.compile(loss=Custom_Loss_Prices(),optimizer='adam')
    else:
        model.compile(loss=loss, optimizer='adam')
    model.summary()                                    
    
    return model

class Custom_Loss_Prices(tf.keras.losses.Loss):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.prices = 7*[1]+5*[5]+12*[1] # 24 hourly "prices" [.1,.1,.1,.1... .3,.3,.3,.3,... 1,1,1] / 20
        self.prices = [self.prices[i//4] for i in range(4*len(self.prices))] # 96 hourly prices
        self.prices_tf = tf.constant(self.prices,dtype=tf.float32)
    def call(self, y_true, y_pred):        
        # elements = tf.multiply(x=self.prices_tf, y=tf.abs(y_true - y_pred))
        # elements = y_true - y_pred # for mse
        elements = tf.multiply(x=y_true, y=y_true - y_pred)
        return tf.reduce_mean(tf.square(elements)) 

def config(plot_theme,seed,precision):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        assert sys.version_info >= (3, 5)

        if plot_theme=='dark':
                plt.style.use('dark_background')
        elif plot_theme=='light':                                
                plt.style.use('seaborn-whitegrid')

        pd.set_option('precision', precision)
        np.random.seed(seed)
        tf.random.set_seed(seed) 
        
def emd_sift(df):
    imf = emd.sift.sift(df['Load'].values)

    for i in range(imf.shape[1]):
            df['IMF%s'%(i+1)] = imf[:,i]    

    return df    

def get_dat(IS_COLAB,site,features,emd=True):

    if site == 'deere':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )
        if emd: 
            df = emd_sift(df)    

    elif site == 'deere-supercleaned':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )
        if emd:
            df = emd_sift(df)

    elif site == 'hyatt':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-10)'],
                                                    index_col=['Datetime (UTC-10)'] )             
        
    elif site == 'lajolla':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-8)'],
                                                    index_col=['Datetime (UTC-8)'] )    

    elif site == 'nwe':
        df = pd.read_csv(     '/content/drive/MyDrive/Data/NWE/ca_actual.csv',     
                                                comment='#',                                 
                                                parse_dates=['Date'],
                                                index_col=['Date'])
        df = convert_nwe_data_to_vector(df)                                                        

                                                                    
    df['Day'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek        

    return df[features]

def get_dat_v2(site,features,emd=True):

    if site == 'deere':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )
        if emd: 
            df = emd_sift(df)    

    elif site == 'deere-supercleaned':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )
        if emd:
            df = emd_sift(df)

    elif site == 'hyatt':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-10)'],
                                                    index_col=['Datetime (UTC-10)'] )             
        
    elif site == 'lajolla':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-8)'],
                                                    index_col=['Datetime (UTC-8)'] )    

    elif site == 'nwe':
        df = pd.read_csv(     '/content/drive/MyDrive/Data/NWE/ca_actual.csv',     
                                                comment='#',                                 
                                                parse_dates=['Date'],
                                                index_col=['Date'])
        df = convert_nwe_data_to_vector(df)

    if df.columns[0] != 'Load':
        df = df.rename(columns={df.columns[0]:'Load'})
                                                                    
    df['Day'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek        
    
    return df[features]        

def get_dat_v3(site,features,emd=True):

    if site == 'deere':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )
        if emd: 
            df = emd_sift(df)    

    elif site == 'deere-supercleaned':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )
        if emd:
            df = emd_sift(df)

    elif site == 'hyatt':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-10)'],
                                                    index_col=['Datetime (UTC-10)'] )             
        
    elif site == 'lajolla':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-8)'],
                                                    index_col=['Datetime (UTC-8)'] )    

    elif site == 'nwe':
        df = pd.read_csv(     '/content/drive/MyDrive/Data/NWE/ca_actual.csv',     
                                                comment='#',                                 
                                                parse_dates=['Date'],
                                                index_col=['Date'])
        df = convert_nwe_data_to_vector(df)
        if emd:
            df = emd_sift(df)

    elif site == 'terna':
        file_path = '/content/drive/MyDrive/Data/terna_load_kw.csv'
        dfm = pd.read_csv(    file_path,
                                                comment='#',
                                                index_col=0)

        idx = pd.date_range(    start     = '2006-1-1 0:00',
                                                    end         = '2015-12-31 23:00',
                                                    freq        = 'H')

        df = pd.DataFrame(index=idx, data=np.empty(len(idx)), columns=['Load'])

        begin, end = 0, 24
        for i in range(dfm.shape[0]):
            dat = dfm.iloc[i].values
            df['Load'].iloc[begin:end] = dat
            begin, end = begin+24, end+24

        # 2006-03-26 02:00:00     NaN
        # 2007-03-25 02:00:00     NaN
        # 2008-03-30 02:00:00     NaN
        # 2009-03-29 02:00:00     NaN
        # 2010-03-28 02:00:00     NaN
        # 2011-03-27 02:00:00     NaN
        # 2012-03-25 02:00:00     NaN
        # 2013-03-31 02:00:00     NaN
        # 2014-03-30 02:00:00     NaN
        # 2015-03-29 02:00:00     NaN
        df = df.fillna(method='ffill')
        df['Load'][df['Load'].isna()]

        if emd:
            df = emd_sift(df)

    if df.columns[0] != 'Load':
        df = df.rename(columns={df.columns[0]:'Load'})
                                                                        
    df['Day'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek        
        
    return df[features]

def get_dat_v4(site,filename=None,features='all',emd=True,rename=False,start=None,end=None):

    if site == 'deere':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )

    elif site == 'deere-supercleaned':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-6)'],
                                                    index_col=['Datetime (UTC-6)'] )

    elif site == 'hyatt':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-10)'],
                                                    index_col=['Datetime (UTC-10)'] )             
        
    elif site == 'lajolla':
        df = pd.read_csv(    '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',     
                                                    comment='#',
                                                    parse_dates=['Datetime (UTC-8)'],
                                                    index_col=['Datetime (UTC-8)'] )    

    elif site == 'nwe':
        df = pd.read_csv(     '/content/drive/MyDrive/Data/NWE/ca_actual.csv',     
                                                comment='#',                                 
                                                parse_dates=['Date'],
                                                index_col=['Date'])
        df = convert_nwe_data_to_vector(df)

    elif site == 'terna':
        file_path = '/content/drive/MyDrive/Data/terna_load_kw.csv'
        dfm = pd.read_csv(    file_path,
                                                comment='#',
                                                index_col=0)

        idx = pd.date_range(    start     = '2006-1-1 0:00',
                                                    end         = '2015-12-31 23:00',
                                                    freq        = 'H')

        df = pd.DataFrame(index=idx, data=np.empty(len(idx)), columns=['Load'])

        begin, end = 0, 24
        for i in range(dfm.shape[0]):
            dat = dfm.iloc[i].values
            df['Load'].iloc[begin:end] = dat
            begin, end = begin+24, end+24

        # 2006-03-26 02:00:00     NaN
        # 2007-03-25 02:00:00     NaN
        # 2008-03-30 02:00:00     NaN
        # 2009-03-29 02:00:00     NaN
        # 2010-03-28 02:00:00     NaN
        # 2011-03-27 02:00:00     NaN
        # 2012-03-25 02:00:00     NaN
        # 2013-03-31 02:00:00     NaN
        # 2014-03-30 02:00:00     NaN
        # 2015-03-29 02:00:00     NaN
        df = df.fillna(method='ffill')
        df['Load'][df['Load'].isna()]
        
    else:
        df = pd.read_csv(filename,
                                         comment='#',
                                         index_col=0,
                                         parse_dates=True)
        
    if rename:
        df = df.rename(columns={df.columns[0]:'Load'})

    if df.columns[0] != 'Load':
        input('\n/// Warning pass rename=True to rename (enter to ack): ')
        #df = df.rename(columns={df.columns[0]:'Load'})
        
    if emd:
        df = emd_sift(df)
        
    if start and end:
        df = df.loc[start:end,:]
                                                                        
    df['Day'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek        
    
    dppd = {'H':24,'15T':96,'T':1440}[df.index.inferred_freq]
        
    d = df['Load'].values.flatten()
    rmse_np1d = rmse(d[(dppd*1):],d[:-(dppd*1)])
    rmse_np7d = rmse(d[(dppd*7):],d[:-(dppd*7)])

    if rmse_np1d < rmse_np7d:
        np_days = 1
    else:
        np_days = 7
    
    if features=='all':
        return df, dppd, np_days
    else:
        return df[features], dppd, np_days

def get_ev_dat(features=['Load']):
    df = pd.read_csv(    '/content/drive/MyDrive/Data/ucsd-ev_load.csv',     
                                                comment='#',
                                                parse_dates=True,
                                                index_col=0) 
    
    df = df[:'2020-2']

    df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'
    df.index.to_series().diff().describe()
    df = df.resample('H').mean()
    df = df.tz_convert(None)
    df = df.fillna(method='ffill')        

    df = emd_sift(df)
                                                                
    df['Day'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek 
    
    return df[features]

def get_bills_dat(features=['Load']):
        df = pd.read_csv('/content/drive/MyDrive/bailey_load.csv',parse_dates=True, index_col=0)
        df = emd_sift(df)
        df['Day'] = df.index.dayofyear
        df['Hour'] = df.index.hour
        df['Weekday'] = df.index.dayofweek        
        return df[features]
    
def get_habitat_dat(features=['Load'],all_features=False):
        df = pd.read_csv('/content/drive/MyDrive/HabitatZEH_60min_processed2_mjw.csv',parse_dates=True, index_col=0)
        df = df[['Load']].loc['2005-12-7':'2022-5-2'] # full days only
        df = emd_sift(df)
        df['Day'] = df.index.dayofyear
        df['Hour'] = df.index.hour
        df['Weekday'] = df.index.dayofweek        
        
        if not all_features: 
                return df[features]        
        else:
                return df    
            
def get_gdrive_dat(filename, cols=['Datetime', 'Load'], features='All'):
        df = pd.read_csv('/content/drive/MyDrive/'+filename,
                                        parse_dates=True, 
                                        index_col=0, 
                                        usecols=cols)
        df.columns = ['Load']

        df = emd_sift(df)
        df['Day'] = df.index.dayofyear
        df['Hour'] = df.index.hour
        df['Weekday'] = df.index.dayofweek

        # measurements per day
        if df.index.inferred_freq == 'T':
                mpd = 60*24
        elif df.index.inferred_freq == '15T':
                mpd = 4*24
        elif df.index.inferred_freq == 'H':
                mpd = 24

        if features == 'All':
                return df, mpd
        else:
                return df[features], mpd        

def convert_nwe_data_to_vector(df):

        # convert matrix of local-time values to vector of standard time values
        vec = []
        for __, row in df.iterrows():
                if np.isnan(row['2nd HR 2']): 
                        for h in range(1,25): 
                                if np.isnan(row['Hr %d'%h]): pass
                                else: vec.append(row['Hr %d'%h])                
                else:
                        for h in range(1,3):
                                if np.isnan(row['Hr %d'%h]): pass
                                else: vec.append(row['Hr %d'%h])
                        vec.append(row['2nd HR 2'])
                        for h in range(3,25):
                                if np.isnan(row['Hr %d'%h]): pass
                                else: vec.append(row['Hr %d'%h])
                        
        nv =    np.asarray(vec)    # list to np array

        for val in nv: 
                if np.isnan(val): print('nan!') # check again for nan's

        # stuff into data frame (could be nice to have datetime index)
        dates = pd.date_range( start=df.index.min(),
                                                        periods=len(vec),
                                                        freq='H')
        df = pd.DataFrame(nv,index=dates,columns=['Load'])                                                        

        return df 

def import_temp_nwe(path):
        filename = path + 'NWE/temp_mt-helena.csv'
        df = pd.read_csv(     filename,     
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

"""
Organize data into batches for training
"""
def organize_dat(df, shift_steps):
    train_split = 0.9
    batch_size = 256
    sequence_length = 96 * 7 * 2    
    
    # shift for forecast
    shift_steps = 1 * 24 * 4    # Number of time steps
    df_targets = df['Load'].shift(-shift_steps)
    
    x_data = df.values[0:-shift_steps]

    y_data = df_targets.values[:-shift_steps]
    y_data = np.expand_dims(y_data,axis=1)

    num_data = len(x_data)
    num_train = int(train_split * num_data)
    num_test = num_data - num_train

    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    len(x_train) + len(x_test)

    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]
    len(y_train) + len(y_test)

    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]

    x_scaler = MinMaxScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    generator = batch_generator(    batch_size,
                                                                sequence_length,
                                                                num_x_signals,
                                                                num_y_signals,
                                                                num_train,
                                                                x_train_scaled,
                                                                y_train_scaled)
    
    x_batch, y_batch = next(generator)

    validation_data = ( np.expand_dims(x_test_scaled, axis=0),
                                            np.expand_dims(y_test_scaled, axis=0))
    
    return (num_x_signals, num_y_signals, generator, validation_data, x_scaler, y_scaler)

"""
Organize data into batches for training
numpy arrays are scaled, pandas dataframes are unscaled
"""
def organize_dat_v2(df, shift_steps=96, sequence_length=96*7*2):
    train_split = 0.9
    batch_size = 256

    # scalers
    feature_scaler = MinMaxScaler()
    load_scaler = MinMaxScaler()
    
    # shift for forecast
    #shift_steps = 1 * 24 * 4    # Number of time steps
    df_targets = df['Load'].shift(-shift_steps)
    
    # scale and adjust the length
    x_data = feature_scaler.fit_transform(df.values)[0:-shift_steps]
    y_data = load_scaler.fit_transform(df_targets.values[:-shift_steps,np.newaxis])
    #y_data = np.expand_dims(y_data,axis=1)

    num_data = len(x_data)
    num_train = int(train_split * num_data)
    num_test = num_data - num_train

    x_train = x_data[0:num_train]
    x_test = x_data[num_train:]
    len(x_train) + len(x_test)

    y_train = y_data[0:num_train]
    y_test = y_data[num_train:]
    len(y_train) + len(y_test)

    num_x_signals = x_data.shape[1]
    num_y_signals = y_data.shape[1]

    generator = batch_generator(    batch_size,
                                                                sequence_length,
                                                                num_x_signals,
                                                                num_y_signals,
                                                                num_train,
                                                                x_train,
                                                                y_train)
    
    x_batch, y_batch = next(generator)

    validation_data = ( np.expand_dims(x_test, axis=0),
                                            np.expand_dims(y_test, axis=0))
    
    return (num_x_signals, num_y_signals, generator, validation_data, load_scaler)



""" Not clear this ever worked
# def organize_dat_v3(df, shift_steps=96, sequence_length=96*7*2, train_split=0.9, 
#                                         batch_size=256, onehot=False):

#         # scalers
#         feature_scaler = MinMaxScaler()
#         load_scaler = MinMaxScaler()

#         # shift for forecast
#         if not onehot:
#             df_targets = df['Load'].shift(-shift_steps)
#         else:
#             df['Load (OH)'] = create_one_hot_vector_of_daily_peak_hr(df[['Load']])
#             df_targets = create_one_hot_vector_of_daily_peak_hr(df[['Load']]).shift(-shift_steps)

#         # scale and adjust the length to remove NaNs caused by .shift()
#         x_data = feature_scaler.fit_transform(df.values)[0:-shift_steps]
#         y_data = load_scaler.fit_transform(df_targets.values[:-shift_steps,np.newaxis])

#         num_data = len(x_data)
#         num_train = int(train_split * num_data)
#         num_test = num_data - num_train

#         x_train = x_data[0:num_train]
#         x_test = x_data[num_train:]
#         len(x_train) + len(x_test)

#         y_train = y_data[0:num_train]
#         y_test = y_data[num_train:]
#         len(y_train) + len(y_test)

#         num_x_signals = x_data.shape[1]
#         num_y_signals = y_data.shape[1]

#         generator = batch_generator(    batch_size,
#                                                                     sequence_length,
#                                                                     num_x_signals,
#                                                                     num_y_signals,
#                                                                     num_train,
#                                                                     x_train,
#                                                                     y_train)

#         x_batch, y_batch = next(generator)

#         validation_data = ( np.expand_dims(x_test, axis=0),
#                                                 np.expand_dims(y_test, axis=0))

#         df = df.iloc[:-shift_steps, :]
#         df = df.iloc[num_train:, :]
#         return (num_x_signals, num_y_signals, generator, validation_data, load_scaler, df)
"""

"""
Generator function for creating random batches of training-data.
"""
def batch_generator(batch_size, sequence_length, num_x_signals, num_y_signals, num_train, x_train_scaled, y_train_scaled):
        
        # Infinite loop.
        while True:
                # Allocate a new array for the batch of input-signals.
                x_shape = (batch_size, sequence_length, num_x_signals)
                x_batch = np.zeros(shape=x_shape, dtype=np.float16)

                # Allocate a new array for the batch of output-signals.
                y_shape = (batch_size, sequence_length, num_y_signals)
                y_batch = np.zeros(shape=y_shape, dtype=np.float16)

                # Fill the batch with random sequences of data.
                for i in range(batch_size):
                        # Get a random start-index.
                        # This points somewhere into the training-data.
                        idx = np.random.randint(num_train - sequence_length) # num_train = 299776
                        
                        # Copy the sequences of data starting at this index.
                        x_batch[i] = x_train_scaled[idx:idx+sequence_length]
                        y_batch[i] = y_train_scaled[idx:idx+sequence_length]
                
                yield (x_batch, y_batch)


def loss_mse_warmup(y_true, y_pred):
        """
        Calculate the Mean Squared Error between y_true and y_pred,
        but ignore the beginning "warmup" part of the sequences.
        
        y_true is the desired output.
        y_pred is the model's output.
        """
        warmup_steps = 50

        # The shape of both input tensors are:
        # [batch_size, sequence_length, num_y_signals].

        # Ignore the "warmup" parts of the sequences
        # by taking slices of the tensors.
        y_true_slice = y_true[:, warmup_steps:, :]
        y_pred_slice = y_pred[:, warmup_steps:, :]

        # These sliced tensors both have this shape:
        # [batch_size, sequence_length - warmup_steps, num_y_signals]

        # Calculat the Mean Squared Error and use it as loss.
        mse = mean(square(y_true_slice - y_pred_slice))
        
        return mse                

    
def train_gru(    num_x_signals, num_y_signals, path_checkpoint, 
                                generator, validation_data, units):
    
    optimizer = RMSprop(learning_rate=1e-3)

    model = Sequential()
    model.add( GRU(    units,
                                     return_sequences=True,
                                     input_shape=(None, num_x_signals,)))
    model.add(Dense(num_y_signals, activation='sigmoid'))

    # dont use, not sure why
    """        
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                                    activation='linear',
                                    kernel_initializer=init))
    """                                        
        
    model.compile(loss=loss_mse_warmup, optimizer=optimizer)
    model.summary()
    
    callback_checkpoint = ModelCheckpoint(    filepath=path_checkpoint,
                                                                                    monitor='val_loss',
                                                                                    verbose=1,
                                                                                    save_weights_only=True,
                                                                                    save_best_only=True)
    
    callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                                                        patience=5,
                                                                                        verbose=1)
    
    callback_tensorboard = TensorBoard( log_dir='./23_logs/',
                                                                            histogram_freq=0,
                                                                            write_graph=False)
    
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                                            factor=0.1,
                                                                            min_lr=1e-4,
                                                                            patience=0,
                                                                            verbose=1)
    
    callbacks = [ callback_early_stopping,
                                callback_checkpoint,
                                callback_tensorboard,
                                callback_reduce_lr]

    model.fit(    x=generator,
                            epochs=20,
                            steps_per_epoch=100,
                            validation_data=validation_data,
                            callbacks=callbacks)         

    return model                         

def evaluate_dat(path_checkpoint, model, x_test_scaled, y_test_scaled):

    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    result = model.evaluate(x=x_test_scaled,y=y_test_scaled)
    #result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
    #                                            y=np.expand_dims(y_test_scaled, axis=0))        
    
    print("loss (test-set):", result)

def predict_dat(path_checkpoint, model):

    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    return model.predict(x_test_scaled)
    
    print("loss (test-set):", result)                

def naive_forecast_mse(y,horizon=96):
    y_pred = y[horizon:]
    y_naive = y[:-horizon]
    mse = np.mean(np.square(y_pred - y_naive))
    return mse    

def adf_test(data):
    result = adfuller(data)

    p = result[1]

    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % p)
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value)) 

    alpha = 0.05

    if p > alpha:
        print('Sample likely not stationary (fail to reject H0)')
    else:
        print('Sample likely stationary (reject H0)') 

def shapiro_wilk_test(data):
    N = 5000

    alpha = 0.05

    n_tests = 0
    n_fails = 0

    begin,end=0,N
    while end < len(data):
        n_tests = n_tests + 1

        # normality test
        stat, p = shapiro(data[begin:end])
        #print('Statistics=%.3f, p=%.3f' % (stat, p))
        
        # interpret
        
        # if p > alpha:             
        #     print('Sample looks Gaussian (fail to reject H0)')
        # else: 
        #     print('Sample does not look Gaussian (reject H0)')
            
        if p < alpha: n_fails = n_fails + 1 

        begin,end = begin + N, end + N    

    print('Ratio of failed tests to total tests: %f' % (n_fails/n_tests))         

def shapiro_wilk_test_v2(data, N=5000, output=True):
    alpha = 0.05

    n_tests = 0
    n_fails = 0

    begin,end=0,N
    while end <= len(data):
        n_tests = n_tests + 1

        stat, p = shapiro(data[begin:end])

        # if p<alpha hypothesis failed, and there is evidence population is not
        # normally distributed    
        if p < alpha: 
            n_fails = n_fails + 1 

        begin,end = begin + N, end + N    

    if output:
        print('Ratio of failed hypotheses (non-normal distributions) to tests: %f' % (n_fails/n_tests))

    return p    

def train_lstm(    num_x_signals, num_y_signals, path_checkpoint, 
                                 generator, validation_data, units, epochs):
    
    optimizer = RMSprop( learning_rate=1e-3 )

    model = Sequential()
    model.add( LSTM(    units,
                                        return_sequences=True,
                                        input_shape=(None, num_x_signals,)))
    model.add( Dense( num_y_signals, activation='sigmoid') )

    # dont use, not sure why
    """        
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                                    activation='linear',
                                    kernel_initializer=init))
    """                                        
        
    model.compile(loss=loss_mse_warmup, optimizer=optimizer)
    model.summary()
    
    callback_checkpoint = ModelCheckpoint(    filepath=path_checkpoint,
                                                                                    monitor='val_loss',
                                                                                    verbose=1,
                                                                                    save_weights_only=True,
                                                                                    save_best_only=True)
    
    callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                                                        patience=5,
                                                                                        verbose=1)
    
    callback_tensorboard = TensorBoard( log_dir='./23_logs/',
                                                                            histogram_freq=0,
                                                                            write_graph=False)
    
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                                            factor=0.1,
                                                                            min_lr=1e-4,
                                                                            patience=0,
                                                                            verbose=1)
    
    callbacks = [ callback_early_stopping,
                                callback_checkpoint,
                                callback_tensorboard,
                                callback_reduce_lr]

    #%%time
    model.fit(    x=generator,
                            epochs=epochs,
                            steps_per_epoch=100,
                            validation_data=validation_data,
                            callbacks=callbacks)         

    return model    

def train_lstm2L(    num_x_signals, num_y_signals, path_checkpoint, 
                                 generator, validation_data, units, epochs):
    
    optimizer = RMSprop(learning_rate=1e-3)

    model = Sequential()
    model.add( LSTM(    units,
                                        return_sequences=True,
                                        input_shape=(None, num_x_signals,)))
    model.add( LSTM(    units,
                                        return_sequences=True) )    
    model.add( Dense( num_y_signals, activation='sigmoid') )    

    # model = Sequential([
    #             LSTM(units=units,return_sequences=True,input_shape=[None,num_x_signals]),
    #             LSTM(units=units),
    #             Dense(num_y_signals, activation='sigmoid'),
    #     ])

    model.compile(loss='mse', optimizer=optimizer)
    model.summary()

    # dont use, not sure why
    """        
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                                    activation='linear',
                                    kernel_initializer=init))
    """                                        
    
    callback_checkpoint = ModelCheckpoint(    filepath=path_checkpoint,
                                                                                    monitor='val_loss',
                                                                                    verbose=1,
                                                                                    save_weights_only=True,
                                                                                    save_best_only=True)
    
    callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                                                        patience=5,
                                                                                        verbose=1)
    
    callback_tensorboard = TensorBoard( log_dir='./23_logs/',
                                                                            histogram_freq=0,
                                                                            write_graph=False)
    
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                                            factor=0.1,
                                                                            min_lr=1e-4,
                                                                            patience=0,
                                                                            verbose=1)
    
    callbacks = [ callback_early_stopping,
                                callback_checkpoint,
                                callback_tensorboard,
                                callback_reduce_lr]

    #%%time
    model.fit(    x=generator,
                            epochs=epochs,
                            steps_per_epoch=100,
                            validation_data=validation_data,
                            callbacks=callbacks)         

    return model        

def lstm_build_train_v2(num_x_signals, num_y_signals, path_checkpoint, 
                                                generator, validation_data, units, epochs, 
                                                layers=1, patience=5, verbose=1,dropout=0.):
    
    model = Sequential()
    model.add( LSTM(    units,
                                        return_sequences=True,
                                        input_shape=(None, num_x_signals,)))
    model.add(Dropout(dropout))
    if layers == 2:
        model.add( LSTM(    units,
                                            return_sequences=True) )
        model.add(Dropout(dropout))    
        
    model.add( Dense( num_y_signals, activation='sigmoid') )    

    model.compile(loss='mse', optimizer='adam')
    model.summary()        
        
    callback_checkpoint = ModelCheckpoint(    filepath=path_checkpoint,
                                                                                    monitor='val_loss',
                                                                                    verbose=verbose,
                                                                                    save_weights_only=True,
                                                                                    save_best_only=True)
    
    callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                                                        patience=patience,
                                                                                        verbose=verbose)
    
    callback_tensorboard = TensorBoard( log_dir='./logs/',
                                                                            histogram_freq=0,
                                                                            write_graph=False)
    
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                                            factor=0.1,
                                                                            min_lr=1e-4,
                                                                            patience=0,
                                                                            verbose=verbose)
    
    callbacks = [ callback_early_stopping,
                                callback_checkpoint,
                                callback_tensorboard,]
                                #callback_reduce_lr]

    hx = model.fit(    x=generator,
                                     epochs=epochs,
                                     steps_per_epoch=100,
                                     validation_data=validation_data,
                                     callbacks=callbacks,
                                     verbose=verbose)         

    return model, hx

def lstm_build_train( num_x_signals, num_y_signals, path_checkpoint, 
                                            generator, validation_data, units, epochs, 
                                            layers=1, patience=5, verbose=1,dropout=0.,
                                            afuncs={'lstm':'relu','dense':'sigmoid'},
                                            loss='mse',metrics=['accuracy']):
    
    model = Sequential()
    model.add( LSTM(    units,
                                        return_sequences=True,
                                        input_shape=(None, num_x_signals,),
                                        activation=afuncs['lstm']))
    model.add(Dropout(dropout))
    if layers == 2:
        model.add( LSTM(    units,
                                            return_sequences=True,
                                            activation=afuncs['lstm']) )
        model.add(Dropout(dropout))    
        
    model.add( Dense( num_y_signals, activation=afuncs['dense']) )    

    model.compile(loss=loss, optimizer='adam',metrics=metrics)
    model.summary()                                    
    
    callback_checkpoint = ModelCheckpoint(    filepath=path_checkpoint,
                                                                                    monitor='val_loss',
                                                                                    verbose=verbose,
                                                                                    save_weights_only=True,
                                                                                    save_best_only=True)
    
    callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                                                        patience=patience,
                                                                                        verbose=verbose)
    
    callback_tensorboard = TensorBoard( log_dir='./logs/',
                                                                            histogram_freq=0,
                                                                            write_graph=False)
    
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                                            factor=0.1,
                                                                            min_lr=1e-4,
                                                                            patience=0,
                                                                            verbose=verbose)
    
    callbacks = [ callback_early_stopping,
                                callback_checkpoint,]
                                #callback_reduce_lr]
                                #callback_tensorboard,]
                                #]

    hx = model.fit(    x=generator,
                                     epochs=epochs,
                                     steps_per_epoch=100,
                                     validation_data=validation_data,
                                     callbacks=callbacks,
                                     verbose=verbose)         

    return model, hx

def train_lstm_v3( num_x_signals, num_y_signals, path_checkpoint, 
                    generator, validation_data, units, epochs, 
                    layers=1, patience=5, verbose=1):
    
    model = Sequential()
    model.add( LSTM(    units,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
    if layers == 2:
        model.add( LSTM(    units,
                                            return_sequences=True) )    
        
    model.add( Dense( num_y_signals, activation='sigmoid') )    

    model.compile(loss='mse', optimizer='adam')
    model.summary()                                    
    
    callback_checkpoint = ModelCheckpoint(    filepath=path_checkpoint,
                                            monitor='val_loss',
                                            verbose=verbose,
                                            save_weights_only=True,
                                            save_best_only=True)
    
    callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                patience=patience,
                                                verbose=verbose)
    
    callback_tensorboard = TensorBoard( log_dir='./logs/',
                                        histogram_freq=0,
                                        write_graph=False)
    
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.1,
                                            min_lr=1e-4,
                                            patience=0,
                                            verbose=verbose)
    
    callbacks = [ callback_early_stopping,
                    callback_checkpoint,
                    callback_tensorboard,]
                    #callback_reduce_lr]

    hx = model.fit( x=generator,
                    epochs=epochs,
                    steps_per_epoch=100,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    verbose=verbose)         

    return model, hx
       

def config_new(plot_theme,seed):
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

def import_data_nwe(path,feature):
        # feature = actual | forecast

        # import raw data from OATI OASIS
        filename = path + 'NWE/ca_'+feature+'.csv'
        df = pd.read_csv(     filename,     
                                                comment='#',                                 
                                                parse_dates=['Date'],
                                                index_col=['Date'])

        # convert matrix of local-time values to vector of standard time values
        vec = []
        for __, row in df.iterrows():
                if np.isnan(row['2nd HR 2']): 
                        for h in range(1,25): 
                                if np.isnan(row['Hr %d'%h]): pass
                                else: vec.append(row['Hr %d'%h])                
                else:
                        for h in range(1,3):
                                if np.isnan(row['Hr %d'%h]): pass
                                else: vec.append(row['Hr %d'%h])
                        vec.append(row['2nd HR 2'])
                        for h in range(3,25):
                                if np.isnan(row['Hr %d'%h]): pass
                                else: vec.append(row['Hr %d'%h])
                        
        nv =    np.asarray(vec)    # list to np array

        for val in nv: 
                if np.isnan(val): print('nan!') # check again for nan's

        # stuff into data frame (could be nice to have datetime index)
        dates = pd.date_range( start=df.index.min(),
                                                        periods=len(vec),
                                                        freq='H')
        df = pd.DataFrame(nv,index=dates,columns=[feature])     

        return df        

def accuracy_of_onehot_matrix(Y_true,Y_pred):
        # matrices must be one hots
        n_correct = []
        n_batches = Y_true.shape[0]
        for i in range(n_batches):
                are_equal = np.array_equal(Y_true[i,:], Y_pred[i,:])
                n_correct.append(are_equal) 
        return sum(n_correct)/n_batches # return a value 0 to 1

def accuracy_of_NON_onehot_wrt_peaks(Y_true,Y_pred):
    # matrix must NOT be one hots
    # returns a value 0 to 1
    # batches are vertical dim, samples are horizontal dim
    n_correct = []
    n_batches = Y_true.shape[0]
    for i in range(n_batches):
            are_equal = np.argmax(Y_true[i,:]) == np.argmax(Y_pred[i,:])
            n_correct.append(are_equal) #
    return sum(n_correct)/n_batches # return a value 0 to 1                 

def calc_accuracy(y_true, y_pred=None, shift=None, output=None):
    """Assumes one-hot"""

    if shift and not y_pred:
        if shift > 0:
            y_true = y_true[shift:]
            y_pred = y_pred[:-shift]

        if shift < 0:
            abs_shift = shift * -1
            y_true = y_true[:-abs_shift]
            y_pred = y_pred[abs_shift:]        

    batches = int(y_true.shape[0]/24)

    y_true = y_true[:(batches*24)].reshape(batches,24).T
    y_pred = y_pred[:(batches*24)].reshape(batches,24).T

    acc = accuracy_of_onehot_matrix(    Y_true = y_true,
                                                                        Y_pred = y_pred)
    
    if output:
        if shift:
            print(f'shift {shift:2d} acc {acc:.2}')
        else:
            print(f'acc {acc:.2}')
    
    return acc

def calc_accuracy_of_two_vectors(y_true, y_pred):
    """ Assumes only one value per day (not 24-per-day one-hot)"""
    return sum(y_true == y_pred)/len(y_true)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def create_one_hot_vector_of_daily_peak_hr(df):
    # create one-hot vector of peaks
    df['peak'] = np.zeros(df.shape[0],dtype=int)
    df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:,0], 'peak'] = 1    

    return df['peak']     

def create_peak_vector(df):
    mL = df.values.reshape(int(len(df)/24),24)
    mLpk = mL.argmax(axis=1)

    idx = pd.date_range(start=min(df.index),
                                            end=max(df.index),
                                            freq='D')

    fake_data = np.empty(len(df))
    fake_data[:] = np.nan

    df['Lpeak'] = fake_data

    df['Lpeak'].loc[idx] = mLpk
    df.head(24)

    df['Lpeak'] = df['Lpeak'].fillna(method='ffill')

    return df['Lpeak']         


    

def plot_weekly_overlaid(df,ppd=96,begin=0,alpha=0.25,
                                                 period_start=None,period_end=None,days_per_week=7):
    ppw = ppd*days_per_week # points per week
    end = begin + days_per_week*ppd
    if not period_start:
        df2 = df['Load']
    else:
        df2 = df[period_start:period_end]['Load']
    t = np.arange(ppw)
    plt.figure(figsize=(20,10))
    while end < df2.shape[0]:
        y = df2.iloc[begin:end].values.flatten()
        plt.plot(t,y,alpha=alpha)
        begin, end = begin + ppw, end + ppw
    plt.title(f'{int(end/ppw)-1} weeks from {period_start} to {period_end}')    
    plt.show()
    
def plot_daily_overlaid(    df,ppd=96,begin=0,alpha=0.25,
                                                    period_start=None,period_end=None):
    end = begin + ppd
    if not period_start:
        df2 = df['Load']
    else:
        df2 = df[period_start:period_end]['Load']
    t = np.arange(ppd)
    plt.figure(figsize=(20,10))
    while end < df2.shape[0]:
        y = df2.iloc[begin:end].values.flatten()
        plt.plot(t,y,alpha=alpha)
        begin, end = begin + ppd, end + ppd
    plt.title(f'{int(end/ppd)-1} weeks from {period_start} to {period_end}')    

def plot_daily_peak_hours(df,month,day_of_week,samples_per_day=24,
                                                    alpha=0.25,xsize=20,ysize=6):
    samples = samples_per_day
    idx = np.logical_and( df.index.month == month, 
                                                df.index.dayofweek == day_of_week)
    d = df[idx].values
    batches = int(len(d)/samples)
    load = d[:(samples*batches)].reshape(batches,samples).T
    peaks = load.argmax(axis=0)

    f = plt.figure(figsize=(xsize,ysize))
    ax1 = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)

    # daily load
    ax1.plot(load,alpha=alpha)
    ax1.plot(load.mean(axis=1),'--k')
    ax1.title.set_text(f'Daily load (peak hr of mean is {load.mean(axis=1).argmax()})')

    # peak hr vs time
    ax2.plot(peaks)
    ax2.title.set_text(f'Peak hr vs time')

    # peak hr distribution
    ax3.hist(peaks, weights=np.ones(len(peaks)) / len(peaks))
    ax3.title.set_text(f'Distribution of peak hr')
    f.gca().yaxis.set_major_formatter(PercentFormatter(1))

    f.show()


def accuracy_one_hot(true,pred):
        """ Measure the accuracy of two one hot vectors, inputs can be 1d numpy or dataseries"""
        n_misses = sum(true != pred)/2         # every miss gives two 'False' entries
        return 1 - n_misses/sum(true)     # basis is the number of one-hots

def one_hot_of_peaks(ds,freq='D'):
        df = pd.DataFrame(ds)
        df['peak'] = 0
        df.loc[df.groupby(pd.Grouper(freq=freq)).idxmax().iloc[:,0], 'peak'] = 1    
        return df['peak']     

def run_the_joules_peak(
                site='prpa',
                units=24,
                layers=1,
                sequence_length=24,
                epochs=100,
                dropout=0,
                patience=10,
                verbose=0,
                output = True,
                plots = False,
                filename = 'data/PRPA_load_cleaned_mjw.csv',
                shift_steps = 1,
                dir = 'models',
                features = [    'Load',
                                'Day',
                                'Weekday',
                                'Hour',
                                'IMF1',                                                                
                                'IMF2',                                                                
                                'IMF3',
                                'IMF4',
                                'IMF5',
                                'IMF6',
                                'IMF7',
                                'IMF8',],
                targets = ['TargetsOH'],
                train_split = 0.9,
                afuncs={'lstm':'relu','dense':'relu','gru':'relu'},
                loss='binary_crossentropy',
                metrics=['accuracy'],
                data_start=None,
                data_end=None):
        
        results = {}

        t = datetime.now()
        path_checkpoint = f'{dir}/{site}/{t.year}-{t.month:02}-{t.day:02}_'+\
                                        f'{t.hour:02}-{t.minute:02}-{t.second:02}_lstm_{units}x{layers}x{shift_steps}.keras'

        df,dppd,np_days = get_dat_v4(site,filename,emd=True,rename=True,start=data_start,end=data_end)                                        

        df['LoadOH'] =            one_hot_of_peaks(df[['Load']])
        df['TargetsOH'] =     one_hot_of_peaks(df[['Load']]).shift(-shift_steps)
        df['PredNPOH'] =        one_hot_of_peaks(df[['Load']]).shift(np_days*dppd-shift_steps)
        df = df.dropna()
        
        # split
        num_data = len(df)
        num_train = int(train_split * num_data)
        df_train = df.iloc[:num_train,:]
        df_valid = df.iloc[num_train:,:]

        feature_scaler = MinMaxScaler()
        X_train = feature_scaler.fit_transform(df_train[features].values)
        X_valid = feature_scaler.fit_transform(df_valid[features].values)

        y_train = df_train.TargetsOH.values[:,np.newaxis]
        y_valid = df_valid.TargetsOH.values[:,np.newaxis]
        
        generator = batch_generator(    batch_size=32,
                                        sequence_length=sequence_length,
                                        num_x_signals=len(features),
                                        num_y_signals=len(targets),
                                        num_train=num_train,
                                        x_train_scaled=X_train,
                                        y_train_scaled=y_train)     
        
        X_batch, y_batch = next(generator)
        
        X_valid = X_valid[np.newaxis,:,:]
        y_valid = y_valid[np.newaxis,:,:] 

        model = Sequential()
        model.add( GRU(    units=units,
                                        return_sequences=True,
                                        input_shape=(None, len(features),),
                                        activation=afuncs['gru']))
        model.add(Dense(len(targets), activation=afuncs['dense']))

        model.compile(loss=loss, optimizer='adam',metrics=metrics)
        model.summary()                                    

        callback_checkpoint = ModelCheckpoint(    filepath=path_checkpoint,
                                                                                        monitor='val_loss',
                                                                                        verbose=verbose,
                                                                                        save_weights_only=True,
                                                                                        save_best_only=True)

        callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                                                        patience=patience,
                                                                                        verbose=verbose)

        callbacks = [ callback_early_stopping,
                                callback_checkpoint,]

        hx = model.fit(    x=generator,
                                epochs=epochs,
                                steps_per_epoch=100,
                                validation_data=(X_valid,y_valid),
                                callbacks=callbacks)                
        
        model.load_weights(path_checkpoint)
        y_valid_pred = model.predict(X_valid)
        
        y_valid_flat            = y_valid[:,:,0].flatten()
        y_valid_pred_flat = y_valid_pred[:,:,0].flatten()
        
        df_valid.loc[:,'y'] = y_valid_flat
        df_valid.loc[:,'y_pred'] = y_valid_pred_flat        
        
        results[f'u{units} sl{sequence_length}'] = accuracy_one_hot(df_valid['y'],one_hot_of_peaks(df_valid['y_pred']))
        

class RunTheJoules:
    def __init__(self,config_file):
        cfg = read_yaml(config_file)
        #self.confg_filepath = config_file
        assert cfg.version == __version__
        self.config = cfg
        self.site = cfg.site
        self.persist_calc_days = cfg.persist_calc_days
        self.data_points_per_day = None
        self.persist_lag = None
        self.results_dir = cfg.results_dir+cfg.site+'/'
        self.clean_dir = cfg.clean_dir
        self.filename = cfg.filename
        self.index_col = cfg.index_col
        self.data_col = cfg.data_col
        self.persist_col = cfg.persist_col
        self.resample = cfg.resample
        self.remove_days=cfg.remove_days # 'weekdays', 'weekdays', or list of ints (0=mon, .., 6=sun)
        self.calendar_features = cfg.calendar_features
        self.model = None
        self.emd = cfg.emd
        self.df = self.get_dat()
        self.peak = self.df['Load'].max()
        self.test_split = cfg.test_split
        self.i_test_split = self.data_points_per_day*int((1-cfg.test_split)*(len(self.df)/self.data_points_per_day))
        self.test_t0 = self.df.index[self.i_test_split]
        self.train = self.df[:self.i_test_split]
        self.valid_split = cfg.valid_split
        self.units_layers = cfg.units_layers
        self.dropout = cfg.dropout
        self.n_in = cfg.n_in
        self.n_out = cfg.n_out
        self.features = cfg.features + [f'{cfg.features_list_name}{i}' for i in cfg.features_list_numbers]
        self.loss = cfg.loss
        self.epochs = cfg.epochs
        self.patience = cfg.patience
        self.plots = cfg.plots
        self.output = cfg.output
        self.verbose = cfg.verbose
        self.test_plots = cfg.test_plots
        self.test_output = cfg.test_output
        self.batch_size = cfg.batch_size
        
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        else:
            if self.clean_dir:
                input('\nWarning: deleting existing files (hit enter): ')
                shutil.rmtree(self.results_dir)
                os.mkdir(self.results_dir)

        shutil.copyfile(os.path.basename(__file__), self.results_dir+os.path.basename(__file__))
        shutil.copyfile(config_file, self.results_dir+config_file)
        
        
    def min_max_scaler(self,df:pd.DataFrame)->pd.DataFrame:
        xmins,xmaxs = [],[]
        for col in df.columns:
            xmin = df[col].minx()
            xmax = df[col].max()
            df[col] = (df[col] - xmin)/(xmax-xmin)
            xmins.append(xmin)
            xmaxs.append(xmax)
            self.scalers = pd.DataFrame({'xmin':xmins,'xmax':xmax}).T
            self.scalers
        return df
        
    def emd_sift(self, df:pd.DataFrame)->pd.DataFrame:
        imf = emd.sift.sift(df['Load'].values)

        for i in range(imf.shape[1]):
                df['IMF%s'%(i+1)] = imf[:,i]    

        return df                  
        
        
    def get_dat(self)->pd.DataFrame:
        usecols = [self.index_col,self.data_col]
        if self.persist_col is not None:
            usecols = usecols + [self.persist_col]
        
        df = pd.read_csv(self.filename,     
                            comment='#',
                            parse_dates=True,
                            index_col=usecols[0],
                            #usecols=usecols
                            )
        
        df = df.ffill().bfill()
        
        if self.resample != False:
            df = df.resample(self.resample).mean()
        
        interval_min = int(df.index.to_series().diff().mode()[0].seconds/60)
        self.data_points_per_day = int(1440/interval_min)
        if self.persist_calc_days:
            self.persist_lag = self.persist_calc_days * self.data_points_per_day
        
        df.columns = [x.split('[')[0].split(' ')[0] for x in df.columns]

        # df = df.rename(columns = {self.data_col:'Load'})
        # if self.persist_col is None:
        #     df['Persist'] = df['Load'].shift(self.persist_lag)
        # else:
        #     df = df.rename(columns = {self.persist_col:'Persist'})
            
        df['weekday'] = df.index.weekday

        #df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'

        #df = df.tz_convert(None)
        df = df.ffill().bfill()
        df = df.ffill().bfill()
        
        
        # if self.resample != False:
        #     df = df.resample(self.resample).mean()
        
        if self.remove_days is not None:
            if self.remove_days == 'weekends':
                df = df[df.index.weekday < 5 ] 
            elif self.remove_days == 'weekdays':
                df = df[df.index.weekday >= 5 ]
            elif isinstance(self.remove_days,list):
                for day in self.remove_days:
                    df = df[df.index.weekday != day]
            else:
                print('remove_days must be "weekends", "weekdays", or a list of integers')
                sys.exit()

        if self.emd:
            df = self.emd_sift(df)
                            
        if self.calendar_features:
            df['Day'] = df.index.dayofyear
            df['Hour'] = df.index.hour
            df['Weekday'] = df.index.dayofweek
        
        #df['Persist'] = df['Load'].shift(self.persist_lag)
    
        df = df.ffill().bfill()

        return df
    
    
    def get_dat_test(self,filename,features):
        print('///// get_dat_test() deprecated ////')
        quit()
        if 0:
            df = pd.read_csv(   filename,     
                                comment='#',
                                parse_dates=True,
                                index_col=0,
                                usecols=[0,self.data_col],)

            df.columns = ['Load']

            #df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'
            df = df.resample('15min').mean()
            #df = df.tz_convert(None)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # df.loc['2020-12-8'] = df['2020-12-1'].values
            # df.loc['2020-11-26'] = df['2020-11-19'].values
            # #df.loc['2021-10-11':'2021-10-15'] = df['2021-10-17':'2021-10-'].values
            # df.loc['2022-5-24'] = df['2022-5-17'].values
            
            
            if self.remove_days == 'weekends':
                df = df[df.index.weekday < 5 ] 
            elif self.remove_days == 'weekdays':
                df = df[df.index.weekday >= 5 ]
            elif isinstance(self.remove_days,list):
                for day in self.remove_days:
                    df = df[df.index.weekday != day]

            df = self.emd_sift(df)
                                    
            if self.calendar_features:                               
                df['Day'] = df.index.dayofyear
                df['Hour'] = df.index.hour
                df['Weekday'] = df.index.dayofweek
            
            df['Persist'] = df['Load'].shift(self.persist_lag)
            
            df = df.ffill().bfill()

            return df[features]
    

    """
    Generator function for creating random batches of training-data.
    """
    def batch_generator_v2(self,
                              batch_size:int,
                              n_in:int,
                              n_out:int,
                              n_x_features:int,
                              n_y_targets:int,
                              n_samples:int,
                              x:int,
                              y:int,
                              randomize=True):
            
            # Infinite loop.
            while True:
                    # Allocate a new array for the batch of input-signals.
                    x_shape = (batch_size, n_in, n_x_features)
                    x_batch = np.zeros(shape=x_shape, dtype=np.float16)

                    # Allocate a new array for the batch of output-signals.
                    y_shape = (batch_size, n_out, n_y_targets)
                    y_batch = np.zeros(shape=y_shape, dtype=np.float16)

                    # Fill the batch with random sequences of data.
                    for i in range(batch_size):
                            # Get a random start-index.
                            # This points somewhere into the training-data.
                            if randomize:
                                idx = np.random.randint(n_samples - n_in - n_in)
                            else:
                                idx = i
                            
                            # Copy the sequences of data starting at this index.
                            x_batch[i] = x[idx:idx+n_in]
                            y_batch[i] = y[idx:idx+n_out]
                    
                    yield (x_batch, y_batch)

    
    def organize_dat_v4(self,
                        df:pd.DataFrame,
                        n_in:int,
                        n_out:int,
                        valid_split:int,
                        batch_size:int,):#shift_steps=96, sequence_length=96*7*2):

        # scalers
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        # shift for forecast
        #shift_steps = 1 * 24 * 4    # Number of time steps
        df_targets = df['Load'].shift(-1*n_in)
        
        # scale and adjust the length
        
        x_data = x_scaler.fit_transform(df.values)[:-1*n_in]
        y_data = y_scaler.fit_transform(df_targets.values[:-1*n_in,np.newaxis]) 
        #y_data = np.expand_dims(y_data,axis=1)
        
        
        dump(x_scaler, open(self.results_dir + "x_scaler.pkl", 'wb'))
        dump(y_scaler, open(self.results_dir + "y_scaler.pkl", 'wb'))

        n_x = len(x_data)
        n_train = int((1-valid_split) * n_x//self.data_points_per_day)*self.data_points_per_day
        #num_valid = num_data - num_train

        x_train = x_data[:n_train]
        x_valid = x_data[n_train:]# if n_train<n_x else None
        #len(x_train) + len(x_valid)

        y_train = y_data[:n_train]
        y_valid = y_data[n_train:]# if n_train<n_x else None
        #len(y_train) + len(y_test)

        n_x_signals = x_data.shape[1]
        n_y_signals = y_data.shape[1]

        train_generator = self.batch_generator_v2(batch_size,
                                            n_in,
                                            n_out, 
                                            n_x_signals,
                                            n_y_signals,
                                            n_train,
                                            x_train,
                                            y_train)

        randomize=False
        n_samples = x_valid.shape[0]
        batch_size = x_valid.shape[0] - n_in - n_out

        x_shape = (batch_size, n_in, n_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, n_out, n_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
                # Get a random start-index.
                # This points somewhere into the training-data.
                if randomize:
                    idx = np.random.randint(n_samples - n_in - n_out)
                else:
                    idx = i
                
                # Copy the sequences of data starting at this index.
                x_batch[i] = x_valid[idx:idx+n_in]
                y_batch[i] = y_valid[idx:idx+n_out]
        
        valid_data = (x_batch,y_batch)

        

        #valid_data = ( np.expand_dims(x_valid, axis=0),np.expand_dims(y_valid, axis=0)) 
        
        return (n_x_signals, n_y_signals, train_generator, valid_data, y_scaler)
    
    def organize_dat_test(self, df:pd.DataFrame):
    
        x_scaler = load(open(self.results_dir + "x_scaler.pkl", 'rb'))
        
        x_test = x_scaler.transform(df.values)

        return np.expand_dims(x_test, axis=0)
    
    def train_lstm_v6(self,
                      n_features_x:int,
                      n_in:int,
                      n_out:int,
                      path_checkpoint:str,
                      train:Generator,
                      valid,
                      units_layers:list,
                      epochs:int, 
                      patience=5,
                      verbose=1,
                      dropout=None,
                      afuncs={'lstm':'relu','dense':'sigmoid'},
                      learning_rate=1e-3,
                      loss='mse',
                      batch_size=None,
                      steps=None):
        
        model = Sequential()
        
        
        #model.add(Input(shape=(n_in,n_features_x)))        
        
        if units_layers[1] == 0:

            model.add( LSTM(units_layers[0],
                            #return_sequences=True,
                            input_shape=(n_in, n_features_x,),
                            activation=afuncs['lstm']) ) 

            if dropout is not None:
                if dropout[0] != 0:
                    model.add(Dropout(dropout[0]))                            

        else:

            model.add( LSTM(units_layers[0],
                            return_sequences=True,
                            input_shape=(n_in, n_features_x,),
                            activation=afuncs['lstm']) ) 

            if dropout is not None:
                if dropout[0] != 0:
                    model.add(Dropout(dropout[0]))                       
        
            model.add( LSTM( units_layers[1],
                            #return_sequences=True,
                            activation=afuncs['lstm']) ) 
            
            if dropout is not None:
                if dropout[1] != 0:
                    model.add(Dropout(dropout[1]))
            
        model.add( Dense( n_out, activation=afuncs['dense']) )    

        if loss == 'custom':
            model.compile(loss=Custom_Loss_Prices(),
                          optimizer=Adam(learning_rate=learning_rate),)
        else:
            model.compile(loss=loss,
                          optimizer=Adam(learning_rate=learning_rate),)
        model.summary()                                    
        
        callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                                monitor='val_loss',
                                                verbose=verbose,
                                                #save_weights_only=True,
                                                save_best_only=True)
        
        callback_early_stopping = EarlyStopping(    monitor='val_loss',
                                                    patience=patience,
                                                    verbose=verbose,
                                                    restore_best_weights=True)
        
        # callback_tensorboard = TensorBoard( log_dir='./logs/',
        #                                     histogram_freq=0,
        #                                     write_graph=False)
        
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.01,
                                                min_lr=1e-4,
                                                patience=5,
                                                verbose=verbose)
        
        # reduce_lr_loss = ReduceLROnPlateau(monitor='loss',
        #                                     factor=0.1,
        #                                     patience=patience_lr,
        #                                     verbose=1,
        #                                     epsilon=1e-4,
        #                                     mode='min')
        
        callbacks = [ callback_early_stopping,
                    callback_checkpoint,
                    #callback_tensorboard,
                    #callback_reduce_lr,
                    ]

        if steps is None:
            steps = (len(self.train) - n_in - n_out)//batch_size

        hx = model.fit( x=train,
                        epochs=epochs,
                        #batch_size=batch_size,
                        steps_per_epoch=steps,
                        validation_data=valid,
                        #validation_split=self.valid_split,
                        callbacks=callbacks,
                        verbose=verbose)

        return model, hx  
    

    def get_callbacks(self, name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        return [mcp_save, reduce_lr_loss]
    
    
    def plot_training_history(self,hx):
        hx_loss = hx.history['loss']
        hx_val_loss = hx.history['val_loss']
        t = np.arange(len(hx_loss))

        plt.figure(figsize=(10,6))
        plt.plot(t,hx_loss,t,hx_val_loss)
        plt.legend(['Train Set Loss','Validation Set Loss'])
        plt.ylabel('MSE (scaled data) [kW/kW]')
        plt.xlabel('Training Epoch')
        plt.title('Training History')     
        plt.show()

    def plot_predictions_day(self,y_true, y_pred, y_pers_wk=None, day=0):
        ppd = self.data_points_per_day
        #ppw = ppd*7 # points per week
        ppp = self.persist_calc_days *ppd
        t = np.arange(ppd)

        y_true_wk = y_true[(day+1)*ppd     :(day+2)*ppd    ]
        y_pred_wk = y_pred[(day+1)*ppd     :(day+2)*ppd    ]
        
        if y_pers_wk is None:
            y_pers_wk = y_true[(day+1)*ppd-ppp :(day+2)*ppd-ppp]

        rmse_pers = np.sqrt(np.mean(np.square(y_pers_wk - y_true_wk)))
        rmse_pred = np.sqrt(np.mean(np.square(y_pred_wk - y_true_wk)))
        skill = (rmse_pers - rmse_pred)/rmse_pers

        plt.figure(figsize=(8,6))
        plt.plot( t,y_true_wk,
                            t,y_pers_wk,
                            t,y_pred_wk)
        
        plt.legend([    'true',
                                    f'persist {self.persist_calc_days}d (rmse {rmse_pers:.0f})', 
                                    f'predict (rmse {rmse_pred:.0f})'])
        
        plt.title(f'test set week {day+1} of {int(len(y_true)/ppd)} '
                            f'(skill {skill:.2})')
        plt.show()    
    
    def plot_predictions_week(self,y_true, y_pers, y_pred, week=0):
        ppd = self.data_points_per_day
        ppw = ppd*7 # points per week
        t = np.arange(ppw)

        y_true_wk = y_true[(week+1)*ppw :(week+2)*ppw]
        y_pers_wk = y_pers[(week+1)*ppw :(week+2)*ppw]
        y_pred_wk = y_pred[(week+1)*ppw :(week+2)*ppw]

        rmse_np1w = np.sqrt(np.mean(np.square(y_pers_wk - y_true_wk)))
        rmse_pred = np.sqrt(np.mean(np.square(y_pred_wk - y_true_wk)))
        skill = (rmse_np1w - rmse_pred)/rmse_np1w

        plt.figure(figsize=(20,6))
        plt.plot( t,y_true_wk,
                            t,y_pers_wk,
                            t,y_pred_wk)
        
        plt.legend([    'true',
                                    f'persist {self.persist_calc_days}d (rmse {rmse_np1w:.0f})', 
                                    f'predict (rmse {rmse_pred:.0f})'])
        
        plt.title(f'test set week {week+1} of {int(len(y_true)/ppw)} '
                            f'(skill {skill:.2})')
        plt.show()
        
        
    def plot_one_prediction(self,model,x_test,y_test,n_in,n_out,begin=0):
        
        y_pred = model.predict(x_test[:,begin:begin+n_in,:])
        
        t_in = np.arange(n_in)
        t_out = np.arange(n_in,n_in+n_out)
        
        for z in range(x_test.shape[2]):
            plt.plot(t_in,x_test[0,begin:begin+n_in,z],label=f'x test f{z}')
        plt.plot(t_out,y_test[0,begin:begin+n_out,0],label='y test')
        plt.plot(t_out,y_pred[0,:n_out,0],'--',label='y pred')
        plt.legend()
        plt.show() 
    
            
    def run_them_fast(self,
                      units_layers:list=None,
                      dropout:list=None,
                      features:str=None,
                      n_in:int=None,
                      n_out:int=None,
                      epochs:int=None,
                      patience:int=None,
                      verbose:int=None,
                      output:bool=None,
                      plots:bool=None,
                      test_split:float=None,
                      valid_split:float=None,
                      batch_size:int=None,
                      loss:str=None):
        
        if units_layers is not None:
            self.units_layers = units_layers
        if dropout is not None:
            self.dropout = dropout
        if features is not None:
            self.features = features
        if n_in is not None:
            self.n_in = n_in
        if n_out is not None:
            self.n_out = n_out
        if epochs is not None:
            self.epochs = epochs
        if patience is not None:
            self.patience = patience
        if plots is not None:
            self.plots = plots
        if output is not None:
            self.output = output
        if verbose is not None:
            self.verbose = verbose
        if test_split is not None:
            self.test_split = test_split
        if valid_split is not None:
            self.valid_split = valid_split
        if batch_size is not None:
            self.batch_size = batch_size
        if loss is not None:
            self.loss = loss

        
        #self.units = units
        #self.layers = layers
        units_layers = self.units_layers
        units = units_layers
        layers = len(units_layers)
        features = self.features
        dropout = self.dropout
        test_split = self.test_split
        valid_split = self.valid_split
        batch_size = self.batch_size
        loss = self.loss
        plots = self.plots
        output = self.output
        verbose = self.verbose
        epochs = self.epochs
        patience = self.patience
        
        df = self.train[self.features]
        
        print(f'\n\n\n////////// units_layers={units_layers} dropout={dropout} n_in={self.n_in} loss={loss}//////////')
        print(f"\n////////// features = {', '.join(features)}//////////\n\n\n")

        # meta
        y, m, d = datetime.now().year-2000, datetime.now().month, datetime.now().day
        path_checkpoint = self.results_dir+ f'lstm.keras'
                
        ( n_features_x, n_features_y, 
            train_gen, valid_data, 
            scaler) = self.organize_dat_v4( df, self.n_in, self.n_out, valid_split, batch_size)

        #(x_valid, y_valid) = dat_valid 

        # np
        #y_valid_naive_mse = naive_forecast_mse( y_valid[0,:,0],horizon=self.persist_lag)

        # model                                                
        self.model, history = self.train_lstm_v6(  n_features_x, self.n_in, self.n_out, 
                                    path_checkpoint, train_gen, valid_data,
                                    units_layers, epochs,
                                    patience, verbose, dropout,loss=loss,batch_size=batch_size)
       #x,y = next(batchgen)

        #y1 = self.model.predict(x[0,:,:])

        #y = self.model.predict(x)

        #y_valid_predict = self.model.predict(x_valid)

        results = {}
        if 0: # not working                                                        
            # evaluate
            y_valid_predict = self.model.predict(x_valid)
            
            y_valid_kw            = scaler.inverse_transform(y_valid[:,:,0]).flatten()
            y_valid_pred_kw = scaler.inverse_transform(y_valid_predict[:,:,0]).flatten()
            y_valid_persist_kw = self.df.iloc[int(valid_split*len(self.df)):,:]['Persist']

            valid_rmse_np = rmse(    y_valid_kw, 
                                    y_valid_persist_kw )
            
            results['valid_rmse_pred']     = rmse(y_valid_kw, y_valid_pred_kw)
            results['valid_skill']         = 1 - results['valid_rmse_pred'] / valid_rmse_np    
            results['valid_std_diff_pred'] = np.diff(y_valid_pred_kw).std()
            results['epochs']              = len(history.history['loss']) - patience
            results['diff_valid_loss']     = pd.DataFrame(history.history['val_loss']).diff().mean()

        if output:
            print('Validation results:')
            print(f'    nRMSE persist {100*valid_rmse_np/self.peak:.1f}%')
            print(f'    nRMSE lstm    {100*rmse(y_valid_kw, y_valid_pred_kw)/self.peak:.1f}%')
            print(f'    Skill (nRMSE) {100 - 100*rmse(y_valid_kw, y_valid_pred_kw)/valid_rmse_np:.1f}%')

        # plot
        if plots is not None:
            if plots == 'hx':
                self.plot_training_history(history)

            if plots == 'valid':
                #for day in range(7):
                    #self.plot_predictions_day(y_valid_kw, y_test_valid_kw, day=day)
                #y_valid_kw = self.df.iloc[int(valid_split*len(self.df)):,:]['Load']
                y_valid_persist_kw = self.df.iloc[int(valid_split*len(self.df)):,:]['Persist']
                self.plot_predictions_week(y_valid_kw, y_valid_persist_kw, y_valid_pred_kw, week=1)
                
                #self.plot_one_prediction(self.model,x_test,y_test,n_in,n_out)

        return results, history

    # def lie_cheat_steal(self,
    #                   #features:str,
    #                   units_layers:list,
    #                   dropout:list,
    #                   #n_in:int,
    #                   #n_out:int,
    #                   epochs=100,
    #                   patience=10,
    #                   verbose=0,
    #                   output=False,
    #                   plots=None,
    #                   valid_split=0.9,
    #                   batch_size=256,
    #                   loss='mse'):
                      
    #     units = units_layers
    #     layers = len(units_layers)

    #     #self.features = features
    #     self.units = units
    #     self.layers = layers
    #     self.dropout = dropout
    #     #self.n_in = n_in
    #     #self.n_out = n_out
    #     self.valid_split = valid_split
    #     self.loss = loss

    #     features = self.features
    #     n_in = self.n_in
    #     n_out = self.n_out
        
    #     df = self.train[features]
        
    #     print(f'\n\n////////// units_layers={units_layers} dropout={dropout} n_in={n_in} loss={loss} //////////\n')

    #     # meta
    #     y, m, d = datetime.now().year-2000, datetime.now().month, datetime.now().day
    #     path_checkpoint = self.results_dir+ f'lstm.keras'
                
    #     ( n_features_x, n_features_y, 
    #         batchgen, dat_valid, 
    #         scaler) = self.organize_dat_v4( df, n_in, n_out, self.train_split, batch_size)

    #     (x_valid, y_valid) = dat_valid 

    #     # np
    #     y_valid_naive_mse = naive_forecast_mse( y_valid[0,:,0],horizon=self.persist_lag)

    #     # model   
    #     #self.model = self.model_builder_kt(h) 
         
    #     tuner = kt.Hyperband(model_builder_kt,
    #                  objective='val_accuracy',
    #                  max_epochs=10,
    #                  factor=3,
    #                  directory='my_dir',
    #                  project_name='intro_to_kt')
                                            
        
    #     callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
    #                                             monitor='val_loss',
    #                                             verbose=verbose,
    #                                             #save_weights_only=True,
    #                                             save_best_only=True)
        
    #     callback_early_stopping = EarlyStopping(    monitor='val_loss',
    #                                                 patience=patience,
    #                                                 verbose=verbose,
    #                                                 restore_best_weights=True)
        
    #     callback_tensorboard = TensorBoard( log_dir='./logs/',
    #                                         histogram_freq=0,
    #                                         write_graph=False)
        
    #     # callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
    #     #                                                                         factor=0.1,
    #     #                                                                         min_lr=1e-4,
    #     #                                                                         patience=0,
    #     #                                                                         verbose=verbose)
        
    #     callbacks = [ callback_early_stopping,
    #                 callback_checkpoint,
    #                 callback_tensorboard,]
    #                 #callback_reduce_lr]

    #     #if steps is None:
    #     steps = (len(self.df) - n_in - n_out)//batch_size

    #     tuner.search(   x=batchgen,
    #                     #label_train,
    #                     epochs=50,
    #                     validation_data=dat_valid,
    #                     validation_split=0.2,
    #                     callbacks=callbacks)

    #     # Get the optimal hyperparameters
    #     best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    #     print(f"""
    #     The hyperparameter search is complete. The optimal number of units in the first densely-connected
    #     layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    #     is {best_hps.get('learning_rate')}.
    #     """)


    #     # hx = model.fit( x=generator,
    #     #                 epochs=epochs,
    #     #                 #batch_size=batch_size,
    #     #                 steps_per_epoch=steps,
    #     #                 validation_data=validation_data,
    #     #                 callbacks=callbacks,
    #     #                 verbose=verbose)  

        

    #     results = {}
    #     if 0: # not working                                                        
    #         # evaluate
    #         y_valid_predict = self.model.predict(x_valid)
            
    #         y_valid_kw            = scaler.inverse_transform(y_valid[:,:,0]).flatten()
    #         y_valid_pred_kw = scaler.inverse_transform(y_valid_predict[:,:,0]).flatten()
    #         y_valid_persist_kw = self.df.iloc[int(valid_split*len(self.df)):,:]['Persist']

    #         valid_rmse_np = rmse(    y_valid_kw, 
    #                                 y_valid_persist_kw )
            
    #         results['valid_rmse_pred']     = rmse(y_valid_kw, y_valid_pred_kw)
    #         results['valid_skill']         = 1 - results['valid_rmse_pred'] / valid_rmse_np    
    #         results['valid_std_diff_pred'] = np.diff(y_valid_pred_kw).std()
    #         results['epochs']             = len(hx.history['loss']) - patience

    #     if output:
    #         print('Validation results:')
    #         print(f'    nRMSE persist {100*valid_rmse_np/self.peak:.1f}%')
    #         print(f'    nRMSE lstm    {100*rmse(y_valid_kw, y_valid_pred_kw)/self.peak:.1f}%')
    #         print(f'    Skill (nRMSE) {100 - 100*rmse(y_valid_kw, y_valid_pred_kw)/valid_rmse_np:.1f}%')

    #     # plot
    #     if plots is not None:
    #         if plots == 'hx':
    #             self.plot_training_history(hx)

    #         if plots == 'valid':
    #             #for day in range(7):
    #                 #self.plot_predictions_day(y_valid_kw, y_test_valid_kw, day=day)
    #             #y_valid_kw = self.df.iloc[int(valid_split*len(self.df)):,:]['Load']
    #             y_valid_persist_kw = self.df.iloc[int(valid_split*len(self.df)):,:]['Persist']
    #             self.plot_predictions_week(y_valid_kw, y_valid_persist_kw, y_valid_pred_kw, week=1)
                
    #             #self.plot_one_prediction(self.model,x_test,y_test,n_in,n_out)

    #     return results, hx.history
    
    def banana_clipper(self,
                       t0:str=None,
                       test_output=None,
                       test_plots=None,
                       limit=None):  
        
        if test_output is not None:
            self.test_output = test_output
        if test_plots is not None:
            self.test_plots = test_plots
        output = self.test_output
        plots = self.test_plots

        if t0 is None:
            t0 = self.test_t0
        else:
            t0 = pd.to_datetime(t0)
        
        if t0 < self.test_t0:
            print(f'Poor form to begin testing on training data, changing start of the test to {self.test_t0}')
            t0 = self.test_t0


        if 'Persist' not in self.df.columns:
            df = self.df[self.features + ['Persist']]
        else:
            df = self.df[self.features]

        y_scaler = load(open(self.results_dir + "y_scaler.pkl", 'rb')) 
        if self.loss == 'custom':
            model = tf.keras.models.load_model(self.results_dir+"lstm.keras",
                                                custom_objects={'Custom_Loss_Prices':Custom_Loss_Prices,})
        else:
            model = tf.keras.models.load_model(self.results_dir+"lstm.keras")
        
        times = []
        skills_mae,maes_pers,maes_lstm = [],[],[]
        skills_mape,mapes_pers,mapes_lstm = [],[],[]
        mases = []
        
        all_forecasts = pd.DataFrame([])
        
        i0 = self.df.index.get_loc(t0)
        hours_remaining = (len(df.index) - i0)//4 - 48
        if limit is not None:
            hours_remaining = limit
        for t in [df.index[i0+4*x] for x in range(hours_remaining)]:
            
            if self.test_output:print(t)

            x = df.loc[:t-pd.Timedelta('15min'),self.features][-1*self.n_in:].copy()
            
            y_true = df.loc[t:,'Load'][:self.n_out].copy()
            y_pers = df.loc[t:,'Persist'][:self.n_out].copy()
            
            mae_pers = (y_true-y_pers).abs().mean()
            mape_pers = ((y_true - y_pers).abs() / y_true).mean()
            
            if mae_pers < 0.001:
                if output:
                    print ('\nPersistence == perfect')
            elif t != y_true.index[0]:
                if output:
                    print ('\nWeekday doesnt exist in training data')
            else:
                x_test = self.organize_dat_test(x)
                
                y_pred = model.predict(x_test[:,-1*self.n_in:,:])
                
                y_pred = pd.Series(y_scaler.inverse_transform(y_pred).flatten(),
                                   index=y_true.index,
                                   name='Pred')
                
                mae_lstm = (y_true - y_pred).abs().mean()
                mape_lstm = ((y_true - y_pred).abs() / y_true).mean()

                skill_mae = 1 - mae_lstm / mae_pers
                skill_mape = 1 - mape_lstm / mape_pers

                mase = (y_true - y_pred).abs().mean() / (y_true - y_pers).abs().mean()
                
                forecast = pd.concat((y_pred,y_true,y_pers),axis=1)

                if plots:
                    title = f'Skill {skill_mae:.3f}'
                    forecast.plot(title=title)
                    plt.show()

                forecast['timestamp_update'] = t
                forecast['timestamp'] = forecast.index
                forecast.index = range(len(all_forecasts),len(all_forecasts)+len(forecast))
                all_forecasts = pd.concat((all_forecasts,
                                           forecast),axis=0)    
                
                times.append(t)
                skills_mae.append(skill_mae)
                maes_pers.append(mae_pers)
                maes_lstm.append(mae_lstm)
                skills_mape.append(skill_mape)
                mapes_pers.append(mape_pers)
                mapes_lstm.append(mape_lstm)
                mases.append(mase)

            
        all_forecasts.to_csv(f'{self.results_dir}/all_forecasts.csv')

        pd.DataFrame({'timestamp_update':times,
                      'skill_mae':skills_mae,
                      'mae_pers':maes_pers,
                      'mae_lstm':maes_lstm,
                      'skill_mape':skills_mape,
                      'mape_pers':mapes_pers,
                      'mape_lstm':mapes_lstm,
                      'mase':mases,}
                      ).round(3).to_csv(f'{self.results_dir}/errors.csv')
                
        skills = np.array(skills_mae)
        print('Percentage of forecasts with positive skill:',
              f'{100*len(skills[skills>0])/len(skills):.0f}%')
        print(f'Average skill: {100*skills.mean():.1f}%')

        return skills.mean(), len(skills[skills>0])/len(skills)
    
    def random_search_warrant(self,units1:list,units2:list,dropout:list,n_in:list,features:list): 
        main_results_dir = self.results_dir
        
        # what are the already completed models?
        search_space_complete = []
        files = next(os.walk(main_results_dir))[1]
        for s in files:
            u1 = int( s.split('_')[0][1:].split('-')[0] )
            u2 = int( s.split('_')[0][1:].split('-')[1] )
            d = float( s.split('_')[1][1:] )
            n = int( s.split('_')[2][2:] )
            flen = int( s.split('_')[3][4:] )
            f = ['Load']
            if flen == 5:
                f = f + [f'IMF{x}' for x in [4,5,6,9]]
            if flen == 13:
                f = f + [f'IMF{x}' for x in range(1,13)]
            search_space_complete.append(dotdict({'u1':u1,'u2':u2,'d':d,'n':n,'f':f}))
                                        
        # build search space
        search_space = []
        for u1 in units1:
            for u2 in units2:
                for d in dropout:
                    for n in n_in:
                        for f in features:
                            search_space.append(dotdict({'u1':u1,'u2':u2,'d':d,'n':n,'f':f}))
        shuffle(search_space)
        
        
        # walk through search space
        results = pd.DataFrame(columns=['units1','units2','dropout','n_in','features','mean_skill',
                                        'positive_skills','epochs'])        
        for s in [x for x in search_space if x not in search_space_complete]: # exlude existing 
            try:                
                self.results_dir = main_results_dir + f'u{s.u1}-{s.u2}_d{s.d}_in{s.n}_flen{len(s.f)}/'
                
                if not os.path.exists(self.results_dir):
                    os.mkdir(self.results_dir)
                _, h = self.run_them_fast(units_layers=[s.u1,s.u2],
                                            dropout=[s.d]*2,
                                            n_in=s.n,
                                            features=s.f)
                mean_skill, positive_skills = self.banana_clipper()
                r = {'mean_skill':mean_skill,'positive_skills':positive_skills,'epochs':len(h['loss'])}
                s.update(r)
                results.loc[len(results)] = s
                results.to_csv(main_results_dir+'/results.csv')
            except:
                pass         

            
if __name__ == '__main__':
     
    model = RunTheJoules('bayfield_jail-courthouse.yaml')

    #r, h = model.run_them_fast()

    #model.banana_clipper(limit=10)  

    model.random_search_warrant([4,8,12,24,48,96,128,256], # units 1
                                [0,4,8,12,24,48,96,128,256], # units 2
                                [0, 0.1],#0,0.1] # dropout
                                [24,48,96,2*96,3*96], # n_in
                                [   ['Load','Persist'], # features
                                    ['Load','Persist','temp'],
                                    ['Load','Persist',]+[f'IMF{x}' for x in [4,5,6,9]],
                                    ['Load','Persist','temp']+[f'IMF{x}' for x in [4,5,6,9]],
                                    #['Load','Persist',]+[f'IMF{x}' for x in range(1,13)],
                                    #['Load','Persist','temp']+[f'IMF{x}' for x in range(1,13)]
                                    ]) 
