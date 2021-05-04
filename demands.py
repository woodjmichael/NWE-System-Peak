# demands.py
# python 3.8.3
# pandas 1.0.5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from _nwe import import_data, find_monthly_peaks, find_daily_peaks, accuracy

print('\nAll times are MST (no DST)')    

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

print(daily_peak_hr['2021-3'].to_string())


print('\n\nNOTE! - Still some bad data (large monthly peaks)\n')






