__version__ = 1.8

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

print('GPU?',tf.config.list_physical_devices('GPU'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#pd.set_option('precision', 2)
pd.options.display.float_format = '{:.2f}'.format
#np.random.seed(42)
#tf.random.set_seed(42) 

class dotdict(dict):
    """ Create a dictionary that can be accessed with dot notation
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def read_yaml(config_file:str) -> dotdict:
    """ Open and parse yaml file into dict

    Args:
        config_file (str): Path to yaml file.

    Returns:
        dotdict: _description_
    """
    with open(config_file, 'r') as stream:
        d=yaml.safe_load(stream)
    cfg = dotdict(d)
    return cfg

def model_builder_kt(hp):
    """ In development

    Args:
        hp (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """ Under development """
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
    """_Define, train, hyperparamter optimze, and test a forecast algorithm based on:
            'Day Ahead Electric Load Forecast: A Comprehensive LSTM-EMD Methodology and Several
            Diverse Case Studies' by M. Wood in MDPI Forecasting (2023)
    """
    def __init__(self,config_file):
        cfg = read_yaml(config_file)
        #self.confg_filepath = config_file
        assert cfg.version == __version__
        self.config = cfg
        self.site = cfg.site
        self.results_dir = cfg.results_dir+cfg.site+'/'
        self.clean_dir = cfg.clean_dir
        
        self.filename = cfg.filename
        self.data_points_per_day = None
        self.index_col = cfg.index_col
        self.target_col = cfg.target_col
        self.benchmark_col = cfg.benchmark_col
        self.usecols = [cfg.index_col, cfg.target_col]
        self.usecols += cfg.benchmark_col if cfg.benchmark_col is not None else []
        self.usecols += cfg.feature_cols if cfg.feature_cols is not None else []
        self.calendar_features = cfg.calendar_features
        self.imf_features = cfg.imf_features
        
        self.benchmark_persist_days = cfg.benchmark_persist_days
        
        self.resample = cfg.resample
        self.remove_days=cfg.remove_days # 'weekdays', 'weekdays', or list of ints (0=mon, .., 6=sun)
        if self.imf_features is not None or self.search:
            self.emd = True
        
        self.df = self.get_dat()
        self.peak = self.df.Load.max()
        self.test_split = cfg.test_split
        self.train_len = int((1-self.test_split) * (len(self.df)/self.data_points_per_day)) \
                        * self.data_points_per_day
        
        self.test_len = len(self.df) - self.train_len
        self.test_t0 = self.df.index[self.train_len]
        self.test_tF = self.df.index[self.train_len+self.test_len-1]
        
        self.i_test_split = self.data_points_per_day*int((1-self.test_split)*(len(self.df)/self.data_points_per_day))
        self.test_t0 = self.df.index[self.i_test_split]
        self.train = self.df[:self.i_test_split]
        
        self.valid_split = cfg.valid_split
        self.units_layers = cfg.units_layers
        self.dropout = cfg.dropout
        self.n_in =  int(cfg.input_length * self.data_points_per_day)
        self.n_out = int(cfg.output_length * self.data_points_per_day)
        self.features = ['Load']
        self.features += cfg.features if cfg.features is not None else []
        self.features += [f'IMF{n}' for n in cfg.imf_features] if cfg.imf_features is not None else []
        self.loss = cfg.loss
        self.epochs = cfg.epochs
        self.patience = cfg.patience
        self.plots = cfg.plots
        self.verbose = cfg.verbose
        self.test_plots = cfg.test_plots
        self.test_output = cfg.test_output
        self.batch_size = cfg.batch_size
        self.forecast_freq = cfg.forecast_freq
        
        self.search = cfg.search
        self.units1 = cfg.units1
        self.units2 = cfg.units2
        self.dropout = cfg.dropout
        self.input_lengths = cfg.input_lengths
        self.features_2D = [['Load']+x for x in cfg.features_2D]
        
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        else:
            if self.clean_dir:
                input('\nWarning: deleting existing files (hit enter): ')
                shutil.rmtree(self.results_dir)
                os.mkdir(self.results_dir)

        shutil.copyfile(os.path.basename(__file__), self.results_dir+os.path.basename(__file__))
        shutil.copyfile(config_file, self.results_dir+config_file)
        
        if cfg.plot_style is not None:
            plt.style.use(cfg.plot_style)
        
        
    def min_max_scaler(self,df:pd.DataFrame)->pd.DataFrame:
        """ Replacement for statstools minmax scaler

        Args:
            df (pd.DataFrame): data to scale

        Returns:
            pd.DataFrame: scaled data
        """
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
        """ Empirical Mode Decomposition receives a single vector of data and decomposes it into
            Intrinsic Mode Functions (IMFs) which typically number in the 10-12 range. Each IMF
            is a component of the original vector and all IMFs summed together necessarily equal the
            original vector. The first IMF is the most rapid oscillation and the last IMF is the 
            "residual" which may have no oscillations. 

        Args:
            df (pd.DataFrame): dataframe containing only one vector of time series data

        Returns:
            pd.DataFrame: original dataframe with new columns for each IMF
        """
        imf = emd.sift.sift(df['Load'].values)

        for i in range(imf.shape[1]):
                df['IMF%s'%(i+1)] = imf[:,i]    

        return df                  
        
        
    def get_dat(self)->pd.DataFrame:
        """ Read and preprocess load data. Resultant data will have a complete index and no bad
            data such as NaNs of inf. Data should be mostly clean to begin with, especially no large
            missing sections.

        Returns:
            pd.DataFrame: data with datetime index (not tz aware)
        """
        df = pd.read_csv(self.filename,     
                            comment='#',
                            parse_dates=True,
                            index_col=self.usecols[0],
                            usecols=self.usecols
                            )
        
        df = df.rename(columns = {self.target_col:'Load'})
        if self.benchmark_col is not None:
            df = df.rename(columns = {self.benchmark_col:'Benchmark'})
        
        df = df.ffill().bfill()
        
        if self.resample != False:
            df = df.resample(self.resample).mean()
        
        interval_min = int(df.index.to_series().diff().mode()[0].seconds/60)
        self.data_points_per_day = int(1440/interval_min)
        
        if self.benchmark_persist_days and self.benchmark_col is None:
            self.persist_lag = self.benchmark_persist_days * self.data_points_per_day
            df['Benchmark'] = df.Load.shift(self.persist_lag)
        
        # drop units from any column headers
        df.columns = [x.split('[')[0] for x in df.columns] # yes these lines are different
        df.columns = [x.split('(')[0] for x in df.columns]
            
        df['Weekday'] = df.index.weekday

        #df = df.tz_localize('Etc/GMT+8',ambiguous='infer') # or 'US/Eastern' but no 'US/Pacific'
        #df = df.tz_convert(None)
        
        df = df.ffill().bfill()        
        
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
    
        df = df.ffill().bfill()

        return df
    
    
    def train_batch_generator( self,
                              batch_size:int,
                              n_in:int,
                              n_out:int,
                              n_x_features:int,
                              n_y_targets:int,
                              n_samples:int,
                              x:int,
                              y:int,
                              randomize=False,
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

    
    def organize_data(self,
                        df:pd.DataFrame,
                        features:list,
                        n_in:int,
                        n_out:int,
                        valid_split:int,
                        batch_size:int,
                        randomize:bool=False):
        """ Structure data for supervised learning

        Args:
            df (pd.DataFrame): data
            features (list): list of features to be used as input  
            n_in (int): input vector length
            n_out (int): output vector length
            valid_split (int): number of values to be used for validation
            batch_size (int): number of input-output pairs per batch

        Returns:
            tuple: (n_x_signals, n_y_signals, train_generator, valid_data, y_scaler)
        """

        # scalers
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        # shift for forecast
        df_features = df[features]
        df_targets = df['Load'].shift(-1*n_in) # confirm that n_in is correct not n_out (2023.10.23)
        
        # scale and adjust the length
        x_data = x_scaler.fit_transform(df_features.values[:-1*n_in]) 
        y_data = y_scaler.fit_transform(df_targets.values[:-1*n_in,np.newaxis])          
        
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

        train_generator = self.train_batch_generator(batch_size,
                                            n_in,
                                            n_out, 
                                            n_x_signals,
                                            n_y_signals,
                                            n_train,
                                            x_train,
                                            y_train)

        
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
    
    def organize_data_test(self, df:pd.DataFrame,features:list):
        """ Organize data for testing

        Args:
            df (pd.DataFrame): data
            features: list of features to be used as input

        Returns:
            np.array: array of x (input) data 
        """
    
        x_scaler = load(open(self.results_dir + "x_scaler.pkl", 'rb'))
        
        x_test = x_scaler.transform(df[features].values)

        return np.expand_dims(x_test, axis=0)
    
    def build_and_train_lstm(self,
                      n_features_x:int,
                      n_in:int,
                      n_out:int,
                      path_checkpoint:str,
                      train:Generator,
                      valid,
                      units_layers:list,
                      epochs:int, 
                      patience=10,
                      verbose=1,
                      dropout:float=None,
                      afuncs={'lstm':'relu','dense':'sigmoid'},
                      learning_rate=1e-3,
                      loss='mse',
                      batch_size=32,
                      steps:int=None):
        """ Define and train LSTM model

        Args:
            n_features_x (int): number of parallel vectors to be used as input
            n_in (int): length of inputs
            n_out (int): length of outputs
            path_checkpoint (str): filename to save best model
            train (Generator): generator object to produce training data
            valid (np.array): validation data
            units_layers (list of int): first number is the 1st layer units, second number is the
                2nd layer units (can be 0 ie no 2nd layer)
            epochs (int): maximum number of epochs to trian for
            patience (int, optional): how many epochs to wait for validation to loss to decrease
                before stopping training. Defaults to 10.
            verbose (int 0-2, optional): how much output information to give while training. Defaults to 1.
            dropout (float, optional): dropout value for each layer. Defaults to None.
            afuncs (dict, optional): activation functions to use for the lstm and ann layers.
                Defaults to {'lstm':'relu','dense':'sigmoid'}.
            learning_rate (float, optional): adam initial learning rate. Defaults to 1e-3.
            loss (str, optional): training loss function. Defaults to 'mse'.
            batch_size (int, optional): number of input-output pairs in each training batch.
                Defaults to 32.
            steps (int, optional): alternative to defining batch_size, can define how many batches
                to divide entire training data into. Defaults to None.

        Returns:
            tuple: (trained tf model, training history)
        """
        
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

        return (model, hx)
    

    def get_callbacks(self, name_weights:str, patience_lr:float):
        """ Callbacks to save best model and reduce learning rate on plateau

        Args:
            name_weights (str): filename for weights
            patience_lr (float): patience learning rate

        Returns:
            list: model checkpoint save, reduced learnign rate
        """
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
        return [mcp_save, reduce_lr_loss]
    
    
    def plot_training_history(self,hx):
        """ Plot training history as epochs vs training loss and validation loss

        Args:
            hx (tf object): history from tf model training
        """
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
    
            
    def organize_data_and_train(self,
                      units_layers:list=None,
                      dropout:list=None,
                      features:str=None,
                      n_in:int=None,
                      n_out:int=None,
                      epochs:int=None,
                      patience:int=None,
                      verbose:int=None,
                      plots:bool=None,
                      test_split:float=None,
                      valid_split:float=None,
                      batch_size:int=None,
                      loss:str=None):
        """ Organize data, train, and save new model. All optional arguments are first defined
            in config file but can be overridden here. 

        Args:
            units_layers (list, optional): Override LSTM layer 1 and 2 (can be 0) units. Defaults to None.
            dropout (list, optional): Override both LSTM layer 1 and 2 dropout. Defaults to None.
            features (str, optional): Override feature list. Defaults to None.
            n_in (int, optional): Override length of input vector. Defaults to None.
            n_out (int, optional): Override length of output vector. Defaults to None.
            epochs (int, optional): Override max number of training epochs. Defaults to None.
            patience (int, optional): Override nubmer of epochs to wait before stopping training. Defaults to None.
            verbose (int 0-2, optional): Override tf training output. Defaults to None.
            plots (bool, optional): Override plots on/off. Defaults to None.
            test_split (float, optional): Override fraction of data for testing . Defaults to None.
            valid_split (float, optional): Override fraction of data for validation. Defaults to None.
            batch_size (int, optional): Override number of input-output pairs for each training batch. Defaults to None.
            loss (str, optional): Override training loss function. Defaults to None.

        Returns:
            tf history object: training history
        """

        units_layers = units_layers if units_layers is not None else self.units_layers
        features = features if features is not None else self.features
        dropout = dropout if dropout is not None else self.dropout
        valid_split = valid_split if valid_split is not None else self.valid_split
        batch_size = batch_size if batch_size is not None else self.batch_size
        loss = loss if loss is not None else self.loss
        plots = plots if plots is not None else self.plots
        verbose = verbose if verbose is not None else self.verbose
        epochs = epochs if epochs is not None else self.epochs
        patience = patience if patience is not None else self.patience
        n_in = n_in if n_in is not None else self.n_in
        n_out = n_out if n_out is not None else self.n_out
        test_split = test_split if test_split is not None else self.test_split
        
        df = self.train[['Benchmark']+features]
        
        print(f'\n\n\n////////// units_layers={units_layers} dropout={dropout} n_in={n_in} loss={loss}//////////')
        print(f"\n////////// features = {', '.join(features)}//////////\n\n\n")

        # meta
        y, m, d = datetime.now().year-2000, datetime.now().month, datetime.now().day
        path_checkpoint = self.results_dir+ f'lstm.keras'
                
        ( n_features_x, n_features_y, 
            train_gen, valid_data, 
            scaler) = self.organize_data( df, features, n_in, n_out, valid_split, batch_size)

        #(x_valid, y_valid) = dat_valid 

        # np
        #y_valid_naive_mse = naive_forecast_mse( y_valid[0,:,0],horizon=self.persist_lag)

        # model                                                
        self.model, history = self.build_and_train_lstm(  n_features_x, n_in, n_out, 
                                    path_checkpoint, train_gen, valid_data,
                                    units_layers, epochs,
                                    patience, verbose, dropout,loss=loss,batch_size=batch_size)

        if plots is not None:
            if plots == 'hx':
                self.plot_training_history(history)

        return history

    
    def test(self,t0:str=None,test_output=None,test_plots=None,limit=None,features=None):  
        """ Quickly test forecast on test data starting at t0

        Args:
            t0 (str, optional): Can specify a starting datetime. Defaults to None.
            test_output (bool, optional): Can turn on/off each test forecast output. Defaults to None.
            test_plots (bool, optional): Can tun on/off each test forecast plot. Defaults to None.
            limit (int, optional): Can limit number of tests. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        if test_output is not None:
            self.test_output = test_output
        if test_plots is not None:
            self.test_plots = test_plots
            
        features = features if features is not None else self.features
        
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
        
        # if 'Persist' not in self.features:
        #     df = self.df[self.features + ['Persist']]
        # else:
        df = self.df[['Benchmark']+self.features]
            
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

            x = df.loc[:t-pd.Timedelta('15min'),self.features+['Benchmark']][-1*self.n_in:].copy()
            
            y_true = df.loc[t:,'Load'][:self.n_out].copy()
            y_pers = df.loc[t:,'Benchmark'][:self.n_out].copy()
            
            mae_pers = (y_true-y_pers).abs().mean()
            mape_pers = ((y_true - y_pers).abs() / y_true).mean()
            
            if mae_pers < 0.001:
                if output:
                    print ('\nPersistence == perfect')
            elif t != y_true.index[0]:
                if output:
                    print ('\nWeekday doesnt exist in training data')
            else:
                x_test = self.organize_data_test(x,features)
                
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
    
    def hyperparam_search(self,
                          units1:list=None,
                          units2:list=None,
                          dropout:list=None,
                          n_in:list=None,
                          features:list=None): 
        """ Random search for best hyperparameters

        Args:
            units1 (list of int): all possible values for the first LSTM layer (cannot be 0)
            units2 (list of int): all possible values for the second LSTM layer (can be 0)
            dropout (list of float): all possible values for dropout (can be 0, typically < 0.3)
            n_in (list of int): all possible lengths of input vector
            features (list of str): all possible combinations of features
        """
        if self.search is not True:
            sys.exit()
            
        if units1 is None: units1 = self.units1
        if units2 is None: units2 = self.units2
        if dropout is None: dropout = self.dropout
        if n_in is None: n_in = [x * self.data_points_per_day for x in self.input_lengths]
        if features is None: features = self.features_2D
            
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
        #shuffle(search_space)
        
        
        # walk through search space
        results = pd.DataFrame(columns=['u1','u2','d','n','fs','mean_skill',
                                        'positive_skills','epochs'])        
        for s in [x for x in search_space if x not in search_space_completed]: # exlude existing 
            try:                
                self.results_dir = main_results_dir + f'u{s.u1}-{s.u2}_d{s.d}_in{s.n}_fs{len(s.f)}/'
                
                if not os.path.exists(self.results_dir):
                    os.mkdir(self.results_dir)
                h = self.organize_data_and_train(units_layers=[s.u1,s.u2],
                                            dropout=[s.d]*2,
                                            n_in=s.n,
                                            features=s.f)
                mean_skill, positive_skills = self.test(features=s.f)
                r = {'mean_skill':mean_skill,'positive_skills':positive_skills,'epochs':len(h.history['loss'])}
                s.update(r)
                s['fs'] = len(s.f)
                results.loc[len(results)] = [s.u1,s.u2,s.d,s.n,s.fs,s.mean_skill,s.positive_skills,s.epochs]
                results.to_csv(main_results_dir+'/results.csv')
                self.analyze_hyperparam_search()
            except:
                pass  
            
    def analyze_hyperparam_search(self):
        """ Calculate best models from hyperparameter search
        """
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
     
    jpl = RunTheJoules('jpl_ev.yaml')
    
    h = jpl.organize_data_and_train()

    jpl.test()
    
    jpl.hyperparam_search()
