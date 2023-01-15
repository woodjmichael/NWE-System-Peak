import pandas as pd
from forecast_tools import *

df = pd.read_csv('data/HabitatZEH_60min_processed2_mjw.csv', parse_dates=True, index_col=0)

df = emd_sift(df)
df['Day'] = df.index.dayofyear
df['Hour'] = df.index.hour
df['Weekday'] = df.index.dayofweek 

(num_x_signals, 
 num_y_signals, 
 generator, 
 validation_data, 
 load_scaler) = organize_dat_v2(df, shift_steps=24, sequence_length=24)

model = Sequential()
model.add( LSTM(1,
                return_sequences=True,
                input_shape=(None, 1)))