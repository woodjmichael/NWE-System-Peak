# nwe_system_peak.py
# python 3.8.3
# pandas 1.0.5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
# funcs
#

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

def import_data(source):
    filename = 'Data/ca_'+source+'.csv'

    df = pd.read_csv(   filename,                    
                        parse_dates=['Date'],
                        index_col=['Date'])#.loc['2007-7-1':'2007-7-2']

    vec = []
    for row in df.iterrows():
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
            
    nv =  np.asarray(vec)

    for val in nv: 
        if np.isnan(val): print('nan!')

    dates = pd.date_range( start=df.index.min(),
                            periods=len(vec),
                            freq='H')

    return pd.DataFrame(nv,index=dates,columns=[source])

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

#
# import and persist
#

df = import_data('actual')['2007-8-1':'2021-3-31']  
df['persist'] = df['actual'].shift(24)
dff = import_data('forecast')['2007-8-1':'2021-3-31']
df = df.dropna()

#
# monthly peak
#

print('\n*\n* monthly peaks \n*\n')

monthly_peaks = find_monthly_peaks(df,dff)

print(monthly_peaks.to_string())
print('\naccuracy (persist)) %: {:.1f}'.format(100*accuracy(monthly_peaks['actual h'],monthly_peaks['persist h'])))
print('accuracy (forecast)) %: {:.1f}'.format(100*accuracy(monthly_peaks['actual h'],monthly_peaks['forecast h'])))


#
# daily peak
#

print('\n*\n* daily peaks \n*\n')

actual = find_daily_peaks(df['actual'])
persist = find_daily_peaks(df['persist'])
forecast = find_daily_peaks(dff['forecast'])

# rearrange for pretty printing
daily_peak_hr = pd.DataFrame(actual['peak hr'].values,columns=['actual'],index=actual.index)
daily_peak_hr['persist'] = persist['peak hr']
daily_peak_hr['forecast'] = forecast['peak hr']

print(daily_peak_hr['2021-1'].to_string())


print('\n\nNOTE!\nStill some bad data (large monthly peaks)\nForecast data may still be bad (check)')






