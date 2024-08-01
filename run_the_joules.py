__version__ = 1.6

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
from tensorflow.keras.backend import square, mean
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, \
                                                                                ReduceLROnPlateau

import emd

print(tf.config.list_physical_devices('GPU'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.style.use('dark_background')#'seaborn-whitegrid')

#pd.set_option('precision', 2)
pd.options.display.float_format = '{:.2f}'.format
#np.random.seed(42)
#tf.random.set_seed(42) 

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
        self.verbose = cfg.verbose
        self.test_plots = cfg.test_plots
        self.test_output = cfg.test_output
        self.batch_size = cfg.batch_size
        self.forecast_freq = cfg.forecast_freq
        
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
            if self.persist_col is None and 'Persist' not in df.columns:
                df['Persist'] = df['Load'].shift(self.persist_lag)
        
        # drop units from any column headers
        df.columns = [x.split('[')[0] for x in df.columns]
        df.columns = [x.split('(')[0] for x in df.columns]

        # df = df.rename(columns = {self.data_col:'Load'})
        # if self.persist_col is None:
        #     df['Persist'] = df['Load'].shift(self.persist_lag)
        # else:
        #     df = df.rename(columns = {self.persist_col:'Persist'})
            
        df['Weekday'] = df.index.weekday

        #df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'

        #df = df.tz_convert(None)
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
    
    
    def batch_generator_v3( self,
                              batch_size:int,
                              n_in:int,
                              n_out:int,
                              n_x_features:int,
                              n_y_targets:int,
                              n_samples:int,
                              x:int,
                              y:int,
                              randomize=True,
                              daily=False):
        """
        Generator function for creating random batches of training-data.

        Args:
            batch_size (int): number of input-vector and output-vector pairs per batch
            n_in (int): lenth of input x vector
            n_out (int): length of output y vector
            n_x_features (int): number of input features
            n_y_targets (int): numer of output targets
            n_samples (int): total number of training set measurements
            x (int): vector of input measurements (shifted ahead by n_in)
            y (int): vector of output measurements
            randomize (bool, optional): option to shuffle each input-output vector pair. Defaults
                to True.
            daily (bool optional): rather than train as if forecast is performed at any arbitrary
                time in the test data, only train for forecasts which are performed exactly at
                midnight.

        Yields:
            _type_: _description_
        """
            
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

        train_generator = self.batch_generator_v3(batch_size,
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

        if plots is not None:
            if plots == 'hx':
                self.plot_training_history(history)

        return history

    
    def banana_clipper(self,
                       t0:str=None,
                       test_output=None,
                       test_plots=None,
                       limit=None,
                       forecast_freq='1h'):  
        
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
            print(f'Bad idea to begin testing on training data, changing start of the test to {self.test_t0}')
            t0 = self.test_t0


        #print(self.features);print(self.df.columns)
        
        if 'Persist' not in self.features:
            df = self.df[self.features + ['Persist']]
        else:
            df = self.df[self.features]
            
        #print(df.columns);sys.exit()

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
        days_remaining = hours_remaining//24
        if limit is not None:
            hours_remaining = limit
        
        if self.forecast_freq == '1h':
            k = 4
            remaining = hours_remaining
        elif self.forecast_freq == '1d':
            k = 96
            remaining = days_remaining
        for t in [df.index[i0+k*x] for x in range(remaining)]:
            
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
        search_space_completed = []
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
            search_space_completed.append(dotdict({'u1':u1,'u2':u2,'d':d,'n':n,'f':f}))
                                        
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
        for s in [x for x in search_space if x not in search_space_completed]: # exlude existing 
            try:                
                self.results_dir = main_results_dir + f'u{s.u1}-{s.u2}_d{s.d}_in{s.n}_flen{len(s.f)}/'
                
                if not os.path.exists(self.results_dir):
                    os.mkdir(self.results_dir)
                h = self.run_them_fast(units_layers=[s.u1,s.u2],
                                            dropout=[s.d]*2,
                                            n_in=s.n,
                                            features=s.f)
                mean_skill, positive_skills = self.banana_clipper()
                r = {'mean_skill':mean_skill,'positive_skills':positive_skills,'epochs':len(h['loss'])}
                s.update(r)
                results.loc[len(results)] = s
                results.to_csv(main_results_dir+'/results.csv')
                model.analyze_hyperparam_search()
            except:
                pass  
            
    def analyze_hyperparam_search(self):
        dirs = next(os.walk(self.results_dir))[1]
        results = pd.DataFrame({'model':[],'mean_skill':[]})
        for d in dirs:
            files = os.listdir(self.results_dir+d)
            if 'errors.csv' in files:
                e = pd.read_csv(self.results_dir+d+'/errors.csv',index_col=1,parse_dates=True)
                mean_skill = e.skill_mae.mean().round(3)
                results.loc[len(results)] = {'model':d,'mean_skill':mean_skill}
        results = results.sort_values(by=['mean_skill'],ascending=False)
        results.to_csv(self.results_dir+'results_summary.csv')
        print(results)
       
       
if __name__ == '__main__':
     
    model = RunTheJoules('jpl_ev.yaml')
    
    h = model.run_them_fast()

    model.banana_clipper()

    # model.random_search_warrant([4,8,12,24,48,96,128,256], # units 1
    #                             [0,4,8,12,24,48,96,128,256], # units 2
    #                             [0, 0.1],#0,0.1] # dropout
    #                             [12,24,48,96,2*96,3*96], # n_in
    #                             [   ['Load','Persist1Workday'], # features
    #                                 #['Load','Persist','temp'],
    #                                 ['Load','Persist1Workday',]+[f'IMF{x}' for x in [3,4]],
    #                                 #['Load','Persist','temp']+[f'IMF{x}' for x in [4,5,6,9]],
    #                                 #['Load','Persist',]+[f'IMF{x}' for x in range(1,13)],
    #                                 #['Load','Persist','temp']+[f'IMF{x}' for x in range(1,13)]
    #                                 ])
    
    #model.analyze_hyperparam_search()
