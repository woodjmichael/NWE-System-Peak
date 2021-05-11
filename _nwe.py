# _nwe.py
# python 3.8.3
# pandas 1.0.5

import pandas as pd
import numpy as np
from numpy import isnan
import matplotlib.pyplot as plt

#
# funcs
# 

def import_data(source, peaks=False):
    filename = 'Data/ca_' + source + '.csv'

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

    if peaks:
        df['peak'] = np.zeros(df.shape[0],dtype=int)

        # one-hot vector denoting peaks 
        df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:,0], 'peak'] = 1

    return df

def find_monthly_peaks(df,dff):

    a = df['actual']
    p = df['persist']
    f = dff['forecast']


    dates = pd.date_range( start=a.index.min(),
                    end=a.index.max(),
                    freq='M')  

    actual,actual_d,actual_h,persist,persist_h = [],[],[],[],[]
    forecast,forecast_h = [],[]

    for date in dates:
        ym = '{}-{}'.format(date.year,date.month)  # year and month string        
        arg = a.loc[ym].argmax() # actual ds
        dt = a.loc[ym].index[arg] # actual ds
        actual.append(a.loc[ym].max())
        actual_d.append(dt.day)
        actual_h.append(dt.hour)

        ymd = '{}-{}-{}'.format(dt.year,dt.month,dt.day)  # year, month, day string (of peak)
        
        arg = p.loc[ymd].argmax() # persist ds
        dt = p.loc[ymd].index[arg] # persist ds
        persist.append(p.loc[ymd].max())
        persist_h.append(dt.hour)

        arg = f.loc[ymd].argmax() # forecast ds
        dt = f.loc[ymd].index[arg] # forecast ds
        forecast.append(f.loc[ymd].max())
        forecast_h.append(dt.hour)        

    df = pd.DataFrame(actual,columns=['actual'],index=dates)
    df['actual d'] = actual_d
    df['actual h'] = actual_h
    df['persist'] = persist
    df['persist h'] = persist_h
    df['forecast'] = forecast
    df['forecast h'] = forecast_h

    return df  

def find_daily_peaks(ds):
    dates = pd.date_range( start=ds.index.min(),
                        end=ds.index.max(),
                        freq='D')                        

    peaks,hrs,times = [], [],  []
    for x in dates:
        ymd = '{}-{}-{}'.format(x.year,x.month,x.day)  # year and month string
        ds_ymd = ds[ymd] # year and month data series
        peaks.append(ds_ymd.max())
        times.append(ds_ymd.index[ds_ymd.argmax()])
        hrs.append(ds_ymd.index[ds_ymd.argmax()].hour)

    df = pd.DataFrame(peaks,columns=['peak'], index=dates)
    df['peak hr'] = hrs
    df['peak time'] = times

    return df     

def rmse(dsA,dsB):
    e = dsA.values - dsB.values
    e2 = np.power(e,2)
    m = np.mean(e2)
    return np.sqrt(m)

def accuracy(dsA,dsB,abs_tol=1):
    hits = 0
    for i in range(len(dsA)):
        if abs(dsA[i]-dsB[i])<abs_tol: hits += 1
    return hits/len(dsA)      