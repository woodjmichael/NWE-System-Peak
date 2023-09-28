import os, sys

from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pprint import pprint

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller # upgrade statsmodels (at least 0.11.1)
from scipy.stats import shapiro, mode

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Embedding, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint, TensorBoard, \
                                        ReduceLROnPlateau
from tensorflow.keras.backend import square, mean

import emd #install using pip

# pd.options.plotting.backend = "plotly"
# pd.set_option('precision', 2)

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

def get_dat(IS_COLAB,site,features,emd=True):

  if site == 'deere':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )
    if emd: 
      df = emd_sift(df)  

  elif site == 'deere-supercleaned':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )
    if emd:
      df = emd_sift(df)

  elif site == 'hyatt':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-10)'],
                          index_col=['Datetime (UTC-10)'] )       
    
  elif site == 'lajolla':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-8)'],
                          index_col=['Datetime (UTC-8)'] )  

  elif site == 'nwe':
    df = pd.read_csv(   '/content/drive/MyDrive/Data/NWE/ca_actual.csv',   
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
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )
    if emd: 
      df = emd_sift(df)  

  elif site == 'deere-supercleaned':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )
    if emd:
      df = emd_sift(df)

  elif site == 'hyatt':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-10)'],
                          index_col=['Datetime (UTC-10)'] )       
    
  elif site == 'lajolla':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-8)'],
                          index_col=['Datetime (UTC-8)'] )  

  elif site == 'nwe':
    df = pd.read_csv(   '/content/drive/MyDrive/Data/NWE/ca_actual.csv',   
                        comment='#',                 
                        parse_dates=['Date'],
                        index_col=['Date'])
    df = convert_nwe_data_to_vector(df)

  if df.columns[0] != 'Load (kW)':
    df = df.rename(columns={df.columns[0]:'Load (kW)'})
                                  
  df['Day'] = df.index.dayofyear
  df['Hour'] = df.index.hour
  df['Weekday'] = df.index.dayofweek    
  
  return df[features]    

def get_dat_v3(site,features,emd=True):

  if site == 'deere':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )
    if emd: 
      df = emd_sift(df)  

  elif site == 'deere-supercleaned':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )
    if emd:
      df = emd_sift(df)

  elif site == 'hyatt':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-10)'],
                          index_col=['Datetime (UTC-10)'] )       
    
  elif site == 'lajolla':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-8)'],
                          index_col=['Datetime (UTC-8)'] )  

  elif site == 'nwe':
    df = pd.read_csv(   '/content/drive/MyDrive/Data/NWE/ca_actual.csv',   
                        comment='#',                 
                        parse_dates=['Date'],
                        index_col=['Date'])
    df = convert_nwe_data_to_vector(df)
    if emd:
      df = emd_sift(df)

  elif site == 'terna':
    file_path = '/content/drive/MyDrive/Data/terna_load_kw.csv'
    dfm = pd.read_csv(  file_path,
                        comment='#',
                        index_col=0)

    idx = pd.date_range(  start   = '2006-1-1 0:00',
                          end     = '2015-12-31 23:00',
                          freq    = 'H')

    df = pd.DataFrame(index=idx, data=np.empty(len(idx)), columns=['Load (kW)'])

    begin, end = 0, 24
    for i in range(dfm.shape[0]):
      dat = dfm.iloc[i].values
      df['Load (kW)'].iloc[begin:end] = dat
      begin, end = begin+24, end+24

    # 2006-03-26 02:00:00   NaN
    # 2007-03-25 02:00:00   NaN
    # 2008-03-30 02:00:00   NaN
    # 2009-03-29 02:00:00   NaN
    # 2010-03-28 02:00:00   NaN
    # 2011-03-27 02:00:00   NaN
    # 2012-03-25 02:00:00   NaN
    # 2013-03-31 02:00:00   NaN
    # 2014-03-30 02:00:00   NaN
    # 2015-03-29 02:00:00   NaN
    df = df.fillna(method='ffill')
    df['Load (kW)'][df['Load (kW)'].isna()]

    if emd:
      df = emd_sift(df)

  if df.columns[0] != 'Load (kW)':
    df = df.rename(columns={df.columns[0]:'Load (kW)'})
                                    
  df['Day'] = df.index.dayofyear
  df['Hour'] = df.index.hour
  df['Weekday'] = df.index.dayofweek    
    
  return df[features]

def get_dat_v4(site,filename=None,features='all',emd=True,rename=False,start=None,end=None):

  if site == 'deere':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )

  elif site == 'deere-supercleaned':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/deere_load_supercleaned.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-6)'],
                          index_col=['Datetime (UTC-6)'] )

  elif site == 'hyatt':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/hyatt_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-10)'],
                          index_col=['Datetime (UTC-10)'] )       
    
  elif site == 'lajolla':
    df = pd.read_csv(  '/content/drive/MyDrive/Data/lajolla_load_IMFs.csv',   
                          comment='#',
                          parse_dates=['Datetime (UTC-8)'],
                          index_col=['Datetime (UTC-8)'] )  

  elif site == 'nwe':
    df = pd.read_csv(   '/content/drive/MyDrive/Data/NWE/ca_actual.csv',   
                        comment='#',                 
                        parse_dates=['Date'],
                        index_col=['Date'])
    df = convert_nwe_data_to_vector(df)

  elif site == 'terna':
    file_path = '/content/drive/MyDrive/Data/terna_load_kw.csv'
    dfm = pd.read_csv(  file_path,
                        comment='#',
                        index_col=0)

    idx = pd.date_range(  start   = '2006-1-1 0:00',
                          end     = '2015-12-31 23:00',
                          freq    = 'H')

    df = pd.DataFrame(index=idx, data=np.empty(len(idx)), columns=['Load (kW)'])

    begin, end = 0, 24
    for i in range(dfm.shape[0]):
      dat = dfm.iloc[i].values
      df['Load (kW)'].iloc[begin:end] = dat
      begin, end = begin+24, end+24

    # 2006-03-26 02:00:00   NaN
    # 2007-03-25 02:00:00   NaN
    # 2008-03-30 02:00:00   NaN
    # 2009-03-29 02:00:00   NaN
    # 2010-03-28 02:00:00   NaN
    # 2011-03-27 02:00:00   NaN
    # 2012-03-25 02:00:00   NaN
    # 2013-03-31 02:00:00   NaN
    # 2014-03-30 02:00:00   NaN
    # 2015-03-29 02:00:00   NaN
    df = df.fillna(method='ffill')
    df['Load (kW)'][df['Load (kW)'].isna()]
    
  else:
    df = pd.read_csv(filename,
                     comment='#',
                     index_col=0,
                     parse_dates=True)
    
  if rename:
    df = df.rename(columns={df.columns[0]:'Load (kW)'})

  if df.columns[0] != 'Load (kW)':
    input('/// Warning pass rename=True to rename (enter to ack): ')
    #df = df.rename(columns={df.columns[0]:'Load (kW)'})
    
  if emd:
    df = emd_sift(df)
    
  if start and end:
    df = df.loc[start:end,:]
                                    
  df['Day'] = df.index.dayofyear
  df['Hour'] = df.index.hour
  df['Weekday'] = df.index.dayofweek    
  
  dppd = {'H':24,'15T':96,'T':1440}[df.index.inferred_freq]
    
  d = df['Load (kW)'].values.flatten()
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

def get_ev_dat(features=['Load (kW)']):
  df = pd.read_csv(  '/content/drive/MyDrive/Data/ucsd-ev_load.csv',   
                        comment='#',
                        parse_dates=True,
                        index_col=0) 
  
  df = df[:'2020-2']

  df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'
  df.index.to_series().diff().describe()
  df = df.resample('H').mean()
  df = df.tz_convert(None)
  df = df.fillna(method='ffill')    

  df = ft.emd_sift(df)
                                
  df['Day'] = df.index.dayofyear
  df['Hour'] = df.index.hour
  df['Weekday'] = df.index.dayofweek 
  
  return df[features]

def get_bills_dat(features=['Load (kW)']):
    df = pd.read_csv('/content/drive/MyDrive/bailey_load.csv',parse_dates=True, index_col=0)
    df = ft.emd_sift(df)
    df['Day'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek    
    return df[features]
  
def get_habitat_dat(features=['Load (kW)'],all_features=False):
    df = pd.read_csv('/content/drive/MyDrive/HabitatZEH_60min_processed2_mjw.csv',parse_dates=True, index_col=0)
    df = df[['Load (kW)']].loc['2005-12-7':'2022-5-2'] # full days only
    df = ft.emd_sift(df)
    df['Day'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Weekday'] = df.index.dayofweek    
    
    if not all_features: 
        return df[features]    
    else:
        return df  
      
def get_gdrive_dat(filename, cols=['Datetime', 'Load (kW)'], features='All'):
    df = pd.read_csv('/content/drive/MyDrive/'+filename,
                    parse_dates=True, 
                    index_col=0, 
                    usecols=cols)
    df.columns = ['Load (kW)']

    df = ft.emd_sift(df)
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
            
    nv =  np.asarray(vec)  # list to np array

    for val in nv: 
        if np.isnan(val): print('nan!') # check again for nan's

    # stuff into data frame (could be nice to have datetime index)
    dates = pd.date_range( start=df.index.min(),
                            periods=len(vec),
                            freq='H')
    df = pd.DataFrame(nv,index=dates,columns=['Load (kW)'])                            

    return df 

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

def emd_sift(df):
  imf = emd.sift.sift(df['Load (kW)'].values)

  for i in range(imf.shape[1]):
      df['IMF%s'%(i+1)] = imf[:,i]  

  return df      

"""
Organize data into batches for training
"""
def organize_dat(df, shift_steps):
  train_split = 0.9
  batch_size = 256
  sequence_length = 96 * 7 * 2  
  
  # shift for forecast
  shift_steps = 1 * 24 * 4  # Number of time steps
  df_targets = df['Load (kW)'].shift(-shift_steps)
  
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

  generator = batch_generator(  batch_size,
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
  #shift_steps = 1 * 24 * 4  # Number of time steps
  df_targets = df['Load (kW)'].shift(-shift_steps)
  
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

  generator = batch_generator(  batch_size,
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

def organize_dat_v3(df, shift_steps=96, sequence_length=96*7*2, train_split=0.9, batch_size=256, onehot=False):

    # scalers
    feature_scaler = MinMaxScaler()
    load_scaler = MinMaxScaler()

    # shift for forecast
    if not onehot:
      df_targets = df['Load (kW)'].shift(-shift_steps)
    else:
      df['Load (OH)'] = create_one_hot_vector_of_daily_peak_hr(df[['Load (kW)']])
      df_targets = create_one_hot_vector_of_daily_peak_hr(df[['Load (kW)']]).shift(-shift_steps)

    # scale and adjust the length to remove NaNs caused by .shift()
    x_data = feature_scaler.fit_transform(df.values)[0:-shift_steps]
    y_data = load_scaler.fit_transform(df_targets.values[:-shift_steps,np.newaxis])

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

    generator = batch_generator(  batch_size,
                                  sequence_length,
                                  num_x_signals,
                                  num_y_signals,
                                  num_train,
                                  x_train,
                                  y_train)

    x_batch, y_batch = next(generator)

    validation_data = ( np.expand_dims(x_test, axis=0),
                        np.expand_dims(y_test, axis=0))

    df = df.iloc[:-shift_steps, :]
    df = df.iloc[num_train:, :]
    return (num_x_signals, num_y_signals, generator, validation_data, load_scaler, df)

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

  
def train_gru(  num_x_signals, num_y_signals, path_checkpoint, 
                generator, validation_data, units):
  
  optimizer = RMSprop(learning_rate=1e-3)

  model = Sequential()
  model.add( GRU(  units,
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
  
  callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
  
  callback_early_stopping = EarlyStopping(  monitor='val_loss',
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

  model.fit(  x=generator,
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
  #                      y=np.expand_dims(y_test_scaled, axis=0))    
  
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
    #   print('Sample looks Gaussian (fail to reject H0)')
    # else: 
    #   print('Sample does not look Gaussian (reject H0)')
      
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

def train_lstm(  num_x_signals, num_y_signals, path_checkpoint, 
                 generator, validation_data, units, epochs):
  
  optimizer = RMSprop( learning_rate=1e-3 )

  model = Sequential()
  model.add( LSTM(  units,
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
  
  callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
  
  callback_early_stopping = EarlyStopping(  monitor='val_loss',
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
  model.fit(  x=generator,
              epochs=epochs,
              steps_per_epoch=100,
              validation_data=validation_data,
              callbacks=callbacks)     

  return model  

def train_lstm2L(  num_x_signals, num_y_signals, path_checkpoint, 
                 generator, validation_data, units, epochs):
  
  optimizer = RMSprop(learning_rate=1e-3)

  model = Sequential()
  model.add( LSTM(  units,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
  model.add( LSTM(  units,
                    return_sequences=True) )  
  model.add( Dense( num_y_signals, activation='sigmoid') )  

  # model = Sequential([
  #       LSTM(units=units,return_sequences=True,input_shape=[None,num_x_signals]),
  #       LSTM(units=units),
  #       Dense(num_y_signals, activation='sigmoid'),
  #   ])

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
  
  callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
  
  callback_early_stopping = EarlyStopping(  monitor='val_loss',
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
  model.fit(  x=generator,
              epochs=epochs,
              steps_per_epoch=100,
              validation_data=validation_data,
              callbacks=callbacks)     

  return model    

def lstm_build_train_v2(num_x_signals, num_y_signals, path_checkpoint, 
                        generator, validation_data, units, epochs, 
                        layers=1, patience=5, verbose=1,dropout=0.):
  
  model = Sequential()
  model.add( LSTM(  units,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
  model.add(Dropout(dropout))
  if layers == 2:
    model.add( LSTM(  units,
                      return_sequences=True) )
    model.add(Dropout(dropout))  
    
  model.add( Dense( num_y_signals, activation='sigmoid') )  

  model.compile(loss='mse', optimizer='adam')
  model.summary()    
    
  callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=verbose,
                                          save_weights_only=True,
                                          save_best_only=True)
  
  callback_early_stopping = EarlyStopping(  monitor='val_loss',
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

  hx = model.fit(  x=generator,
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
  model.add( LSTM(  units,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    activation=afuncs['lstm']))
  model.add(Dropout(dropout))
  if layers == 2:
    model.add( LSTM(  units,
                      return_sequences=True,
                      activation=afuncs['lstm']) )
    model.add(Dropout(dropout))  
    
  model.add( Dense( num_y_signals, activation=afuncs['dense']) )  

  model.compile(loss=loss, optimizer='adam',metrics=metrics)
  model.summary()                  
  
  callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=verbose,
                                          save_weights_only=True,
                                          save_best_only=True)
  
  callback_early_stopping = EarlyStopping(  monitor='val_loss',
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

  hx = model.fit(  x=generator,
                   epochs=epochs,
                   steps_per_epoch=100,
                   validation_data=validation_data,
                   callbacks=callbacks,
                   verbose=verbose)     

  return model, hx

def train_lstm_v3(  num_x_signals, num_y_signals, path_checkpoint, 
                    generator, validation_data, units, epochs, 
                    layers=1, patience=5, verbose=1):
  
  model = Sequential()
  model.add( LSTM(  units,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
  if layers == 2:
    model.add( LSTM(  units,
                      return_sequences=True) )  
    
  model.add( Dense( num_y_signals, activation='sigmoid') )  

  model.compile(loss='mse', optimizer='adam')
  model.summary()                  
  
  callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=verbose,
                                          save_weights_only=True,
                                          save_best_only=True)
  
  callback_early_stopping = EarlyStopping(  monitor='val_loss',
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

  hx = model.fit(  x=generator,
                   epochs=epochs,
                   steps_per_epoch=100,
                   validation_data=validation_data,
                   callbacks=callbacks,
                   verbose=verbose)     

  return model, hx

def train_lstm_v4(  num_x_signals:int, num_y_signals:int, path_checkpoint:str, 
                    generator, validation_data, units:list, epochs:int, 
                    layers:int=1, patience:int=5, verbose:int=1,dropout:list=None,
                    afuncs={'lstm':'relu','dense':'sigmoid'},
                    loss='mse',):
  
  model = Sequential()
  # model.add( LSTM(  units,
  #                   return_sequences=True,
  #                   input_shape=(None, num_x_signals,)))
  model.add( LSTM(units[0],
                  return_sequences=True,
                  input_shape=(None, num_x_signals,),
                  activation=afuncs['lstm']) )
  if dropout is not None:
    model.add(Dropout(dropout[0]))
  if (layers == 2) and (len(units)>1):
    # model.add( LSTM(  units,
    #                   return_sequences=True) ) 
    model.add( LSTM(  units[1],
                  return_sequences=True,
                  activation=afuncs['lstm']) ) 
    if dropout is not None:
      model.add(Dropout(dropout[1]))
    
  model.add( Dense( num_y_signals, activation='sigmoid') )  

  model.compile(loss=loss, optimizer='adam')
  model.summary()                  
  
  callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=verbose,
                                          save_weights_only=True,
                                          save_best_only=True)
  
  callback_early_stopping = EarlyStopping(  monitor='val_loss',
                                            patience=patience,
                                            verbose=verbose,
                                            restore_best_weights=True)
  
  callback_tensorboard = TensorBoard( log_dir='./logs/',
                                      histogram_freq=0,
                                      write_graph=False)
  
  # callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
  #                                     factor=0.1,
  #                                     min_lr=1e-4,
  #                                     patience=0,
  #                                     verbose=verbose)
  
  callbacks = [ callback_early_stopping,
                callback_checkpoint,
                callback_tensorboard,]
                #callback_reduce_lr]

  hx = model.fit(  x=generator,
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
    df = pd.read_csv(   filename,   
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
            
    nv =  np.asarray(vec)  # list to np array

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

  acc = accuracy_of_onehot_matrix(  Y_true = y_true,
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

def plot_predictions_week(y_true, y_pred, week=0, ppd=96):

  ppw = ppd*7 # points per week, points per day
  t = np.arange(ppw)

  y_true_wk = y_true[(week+1)*ppw:(week+2)*ppw]
  y_np1w_wk = y_true[(week)*ppw  :(week+1)*ppw]
  y_pred_wk = y_pred[(week+1)*ppw:(week+2)*ppw]

  rmse_np1w = np.sqrt(np.mean(np.square(y_np1w_wk - y_true_wk)))
  rmse_pred = np.sqrt(np.mean(np.square(y_pred_wk - y_true_wk)))
  skill = (rmse_np1w - rmse_pred)/rmse_np1w

  plt.figure(figsize=(20,6))
  plt.plot( t,y_true_wk,
            t,y_np1w_wk,
            t,y_pred_wk)
  
  plt.legend([  'true',
                f'np 1 wk (rmse {rmse_np1w:.0f})', 
                f'predict (rmse {rmse_pred:.0f})'])
  
  plt.title(f'test set week {week+1} of {int(len(y_true)/ppw)} '
            f'(skill {skill:.2})')
  plt.show()   

def plot_weekly_overlaid(df,ppd=96,begin=0,alpha=0.25,
                         period_start=None,period_end=None):
  ppw = ppd*7 # points per week
  end = begin + 7*ppd
  if not period_start:
    df2 = df['Load (kW)']
  else:
    df2 = df[period_start:period_end]['Load (kW)']
  t = np.arange(ppw)
  plt.figure(figsize=(20,10))
  while end < df2.shape[0]:
    y = df2.iloc[begin:end].values.flatten()
    plt.plot(t,y,alpha=alpha)
    begin, end = begin + ppw, end + ppw
  plt.title(f'{int(end/ppw)-1} weeks from {period_start} to {period_end}')  
  
def plot_daily_overlaid(  df,ppd=96,begin=0,alpha=0.25,
                          period_start=None,period_end=None):
  end = begin + ppd
  if not period_start:
    df2 = df['Load (kW)']
  else:
    df2 = df[period_start:period_end]['Load (kW)']
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
  
def plot_training_history(hx):
  hx_loss = hx.history['loss']
  hx_val_loss = hx.history['val_loss']
  t = np.arange(len(hx_loss))

  plt.figure(figsize=(10,6))
  plt.plot(t,hx_loss,t,hx_val_loss)
  plt.legend(['Train Set Loss','Test Set Loss'])
  plt.ylabel('MSE (scaled data) [kW/kW]')
  plt.xlabel('Training Epoch')
  plt.title('Training History')   

def accuracy_one_hot(true,pred):
    """ Measure the accuracy of two one hot vectors, inputs can be 1d numpy or dataseries"""
    n_misses = sum(true != pred)/2     # every miss gives two 'False' entries
    return 1 - n_misses/sum(true)   # basis is the number of one-hots

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
        features = [  'Load (kW)',
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

    df['LoadOH'] =      one_hot_of_peaks(df[['Load (kW)']])
    df['TargetsOH'] =   one_hot_of_peaks(df[['Load (kW)']]).shift(-shift_steps)
    df['PredNPOH'] =    one_hot_of_peaks(df[['Load (kW)']]).shift(np_days*dppd-shift_steps)
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
    
    generator = batch_generator( batch_size=32,
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
    model.add( GRU(  units=units,
                    return_sequences=True,
                    input_shape=(None, len(features),),
                    activation=afuncs['gru']))
    model.add(Dense(len(targets), activation=afuncs['dense']))

    model.compile(loss=loss, optimizer='adam',metrics=metrics)
    model.summary()                  

    callback_checkpoint = ModelCheckpoint(  filepath=path_checkpoint,
                                            monitor='val_loss',
                                            verbose=verbose,
                                            save_weights_only=True,
                                            save_best_only=True)

    callback_early_stopping = EarlyStopping(  monitor='val_loss',
                                            patience=patience,
                                            verbose=verbose)

    callbacks = [ callback_early_stopping,
                callback_checkpoint,]

    hx = model.fit(  x=generator,
                epochs=epochs,
                steps_per_epoch=100,
                validation_data=(X_valid,y_valid),
                callbacks=callbacks)        
    
    model.load_weights(path_checkpoint)
    y_valid_pred = model.predict(X_valid)
    
    y_valid_flat      = y_valid[:,:,0].flatten()
    y_valid_pred_flat = y_valid_pred[:,:,0].flatten()
    
    df_valid.loc[:,'y'] = y_valid_flat
    df_valid.loc[:,'y_pred'] = y_valid_pred_flat    
    
    results[f'u{units} sl{sequence_length}'] = accuracy_one_hot(df_valid['y'],one_hot_of_peaks(df_valid['y_pred']))
    

if __name__ == '__main__':
  print('banana clipper!')