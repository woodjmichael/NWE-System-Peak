"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           config                                               #
#                                                                                                #
#                                                                                                #
###############################################################################################"""
from forecast import *
IS_COLAB = config(plot_theme='light',seed=42) 

site = 'nwe' # lajolla, hyatt, nwe   
forecast_type = 'peak' # peak or ''
additional_features = [] # IMFx or []
b = 2521 # batches: nwe 2 day =  2521, lajolla 2 day = 377, hyatt 2 day = 528
n = 24 # number of timesteps (input)
h = 0 # horizon of forecast
o = 24 # output timesteps
units = [5] 
epochs = [50]
s1, s2 = int(.63*b), int(.9*b) # split 1 (train-valid), 2 (valid-test)



"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           data                                                 #
#                                                                                                #
#                                                                                                #
###############################################################################################"""

# load
vL = get_data(site,IS_COLAB,additional_features) # load vector
mL = batchify_single_series(vL,b,n+o) # load matrix
Lmax = np.max(mL) 
mL = mL / Lmax # scale by max
d = mL # shape: (batches,timesteps)
if len(additional_features) > 0:
    vIMF3 = get_data(site,IS_COLAB,additional_features) # load vector
    mIMF3 = batchify_single_series(vIMF3,b,n+o) # peaks matrix
    mIMF3 = mIMF3/np.max(mIMF3) # scale by max
    d = np.concatenate((mL,mIMF3),axis=2) # shape: (batches,timesteps,features)

# peaks
if forecast_type == 'peak':
    vP = get_data(site,IS_COLAB,additional_features,onehot_peak=True)
    mP = batchify_single_series(vP,b,n+o) # peaks matrix
    d = np.concatenate((mP,mL),axis=2) # shape: (batches,timesteps,features)
    

# X and y, split train-test-valid
X_train, y_train = d[:s1,   :n-h, :], d[:s1,   -o:, 0] # (features, targets)
X_valid, y_valid = d[s1:s2, :n-h, :], d[s1:s2, -o:, 0]
X_test,  y_test  = d[s2:,   :n-h, :], d[s2:,   -o:, 0]
y_valid_data = d[s1:s2, -o:, -1]

# nwe existing forecast
print_inputs(X_train, y_train, b, n, h,o,units,epochs,forecast_type,site,additional_features)

"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           models                                               #
#                                                                                                #
#                                                                                                #
###############################################################################################"""

res, y_valid_pred, hx, = {}, {}, {}

# naive persistence
y_valid_pred['np'] = naive_persistence(X_valid, y_valid, o, Lmax)[0]

# nwe inhouse/internal forecast
if site == 'nwe':
    res['in-house'], y_valid_pred['in-house'] = nwe_inhouse_forecast(X_valid, y_valid, b, n, o, s1, s2, Lmax, forecast_type, IS_COLAB)

# linear regression
#if fcast == 'normal': # can't use linear_regression() w/ multiple features (peaks, emd)
#    res['reg'], y_valid_pred['reg'], hx['reg'] = linear_regression(1000, X_train, y_train, X_valid, y_valid, n, h, o)

for u in units:
    for e in epochs:
        #m = 'rnn u{} e{}'.format(u,e) # model name        
        #res[m], y_valid_pred[m], hx[m] = deep_rnn(e, X_train, y_train, X_valid, y_valid, n, h, o, u, fcast)

        m = 'lstm u{} e{}'.format(u,e) # model name
        res[m],y_valid_pred[m],hx[m] = lstm(e, X_train, y_train, X_valid, y_valid, n, h, o, u, forecast_type, Lmax)        



"""###############################################################################################
#                                                                                                #
#                                                                                                #
#                                           results                                              #
#                                                                                                #
#                                                                                                #
###############################################################################################"""

print_results(res, X_valid, y_valid, o, Lmax, forecast_type, np_only=False)

plot_predictions(X_valid, y_valid, y_valid_data, y_valid_pred, n, h, o, forecast_type, Lmax, batch=0, title='validation set')

#for b in range(10):
#    plot_predictions(X_valid, y_valid, y_valid_data, y_valid_pred, n, h, o, forecast_type, Lmax, batch=b, title='validation set')

plot_training(hx, first_epoch=25)    