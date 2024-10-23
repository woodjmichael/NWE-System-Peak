# *Load LSTM Forecast*

# **Dev**

## 2021

### May

### 6 - NWE system peak

- dilemma: simple RNN trains in ~20 epochs according to the training history MSE, but validation loss continues to drop through 100, 1000, and 10000 epochs

```
Batches 2020
Input dimension 36
Forecast horizon 0
Output dimension 24
X_train shape (1616, 36, 1)
y_train shape (1616, 24)

rmse {'np': 60.00919, 'reg': 52.638226, 'rnn25': 122.52051, 'rnn100': 82.759056, 'rnn1000': 45.996723, 'rnn10000': 41.400455} 
```

- sequence to sequence isn't going so well, MSE should be closer to 0.006

```
Batch size 50
Forecast horizon 0
Model output size 10
X_train shape (7000, 50, 1)
y_train shape (7000, 10)
y_train_s2s shape (7000, 50, 10)

mse {'naive': 0.46083385, 'reg': 0.019046158, 'rnn': 0.009033798, 'rnn_s2s': 0.01177070737720724} 
```

### 7 - NWE forecast

- lstm is ostensibly working

```
(base) mjw@matacino NWE system peak % python forecast.py

Batches 2020
Input dimension 36
Forecast horizon 0
Output dimension 24
RNN units 50
X_train shape (1616, 36, 1)
y_train shape (1616, 24)

              reg      rnn  lstm 25 lstm 100  lstm 500 lstm 1000
epochs       1000     1000       25      100       500      1000
units           0       50       50       50        50        50
skill_np  6.00779  9.87337 -71.2666 -7.04581  0.102032    6.2431
(base) mjw@matacino NWE system peak % 
```

- still not sure why  model losss stops decreasing after epoch ~25 but skill continues to increase
- ah! model loss IS decreasing (below for lstm 25 epochs, 100, 500, and 1000), as model.evaluate() shows
- remember that model loss (MSE is on the normalized data, so units are super different than RMSE on the real vs predictions)

```
/7 [==============================] - 0s 9ms/step - loss: 5.2279e-05
lstm eval 5.2279392548371106e-05
7/7 [==============================] - 0s 7ms/step - loss: 3.9236e-05
lstm eval 3.923612166545354e-05
7/7 [==============================] - 0s 7ms/step - loss: 1.4439e-05
lstm eval 1.4438621292356402e-05
7/7 [==============================] - 0s 17ms/step - loss: 9.3310e-06
lstm eval 9.331025466963183e-06
              reg  lstm 25 lstm 100 lstm 500 lstm 1000
epochs       1000       25      100      500      1000
units           0       50       50       50        50
skill_np  4.95371 -69.0254 -51.7759 -7.80237    5.4955
```

### 8  - nwe forecast

- run 1: 96 in 24 out

```
(base) mjw@matacino NWE system peak % python forecast.py

Batches 1010
Input dimension 96
Forecast horizon 0
Output dimension 24
Units 50
X_train shape (808, 96, 1)
y_train shape (808, 24)

4/4 [==============================] - 0s 11ms/step - loss: 6.4416e-06
lstm eval 6.441602181439521e-06
4/4 [==============================] - 0s 10ms/step - loss: 6.8023e-06
lstm eval 6.802289135521278e-06
4/4 [==============================] - 0s 11ms/step - loss: 4.7566e-06
lstm eval 4.756640464620432e-06
4/4 [==============================] - 0s 11ms/step - loss: 5.5163e-06
lstm eval 5.51634184375871e-06
4/4 [==============================] - 0s 12ms/step - loss: 5.3757e-06
lstm eval 5.375724867917597e-06
4/4 [==============================] - 0s 14ms/step - loss: 5.3982e-06
lstm eval 5.398239409259986e-06
4/4 [==============================] - 0s 15ms/step - loss: 5.0687e-06
lstm eval 5.068661266705021e-06
4/4 [==============================] - 0s 18ms/step - loss: 5.5383e-06
lstm eval 5.538331151910825e-06
4/4 [==============================] - 0s 19ms/step - loss: 4.6994e-06
lstm eval 4.69935912406072e-06
4/4 [==============================] - 0s 20ms/step - loss: 4.4116e-06
lstm eval 4.4116482058598194e-06
         epochs units skill_np
reg        1000     0  4.46864
rnn 10     5000    10  17.1743
lstm 10    5000    10  13.2812
rnn 20     5000    20  21.7565
lstm 20    5000    20  12.0304
rnn 30     5000    30  20.4333
lstm 30    5000    30  19.6533
rnn 40     5000    40  23.9708
lstm 40    5000    40  16.6602
rnn 50     5000    50  24.2267
lstm 50    5000    50  17.1979
rnn 60     5000    60  21.8431
lstm 60    5000    60  17.1114
rnn 70     5000    70  19.7044
lstm 70    5000    70   18.397
rnn 80     5000    80  24.2441
lstm 80    5000    80  16.5768
rnn 90     5000    90   27.408
lstm 90    5000    90  19.8884
rnn 100    5000   100  22.3785
lstm 100   5000   100  21.0913
```

- run 2: 36 in 24 out

```
(base) mjw@matacino NWE system peak % python forecast.py

Batches 2020
Input dimension 36
Forecast horizon 0
Output dimension 24
Units 50
X_train shape (1616, 36, 1)
y_train shape (1616, 24)

7/7 [==============================] - 0s 6ms/step - loss: 6.7373e-06
lstm eval 6.737269359291531e-06
7/7 [==============================] - 0s 5ms/step - loss: 6.4889e-06
lstm eval 6.488948656624416e-06
7/7 [==============================] - 0s 5ms/step - loss: 5.3429e-06
lstm eval 5.342918029782595e-06
7/7 [==============================] - 0s 6ms/step - loss: 5.5203e-06
lstm eval 5.520279046322685e-06
7/7 [==============================] - 0s 7ms/step - loss: 6.5623e-06
lstm eval 6.56232896290021e-06
7/7 [==============================] - 0s 7ms/step - loss: 5.6421e-06
lstm eval 5.642079031531466e-06
7/7 [==============================] - 0s 21ms/step - loss: 5.1736e-06
lstm eval 5.17361331731081e-06
7/7 [==============================] - 0s 10ms/step - loss: 4.8473e-06
lstm eval 4.847257969231578e-06
7/7 [==============================] - 0s 11ms/step - loss: 4.8379e-06 ** lowest loss **
lstm eval 4.837921096623177e-06
7/7 [==============================] - 0s 10ms/step - loss: 6.7127e-06
lstm eval 6.7126834437658545e-06
         epochs units skill_np
reg        1000     0  5.85384
rnn 10     5000    10  13.6938
lstm 10    5000    10  13.6877
rnn 20     5000    20  16.4120
lstm 20    5000    20  14.5493
rnn 30     5000    30  16.5726
lstm 30    5000    30  18.7586
rnn 40     5000    40  15.6508
lstm 40    5000    40  18.0795
rnn 50     5000    50  17.7428
lstm 50    5000    50   14.293
rnn 60     5000    60  15.8321
lstm 60    5000    60  17.6195
rnn 70     5000    70  16.6178
lstm 70    5000    70  19.4174
rnn 80     5000    80  14.5923
lstm 80    5000    80  20.7186
rnn 90     5000    90  17.4708
lstm 90    5000    90  20.7564  ** highest skill **
rnn 100    5000   100  12.3035
lstm 100   5000   100  13.7723
```

### 9 - nwe forecast

- 24 in, 24 out

```
(base) mjw@matacino NWE system peak % python forecast.py

Batches 2525
Input dimension 24
Forecast horizon 0
Output dimension 24
Units 50
X_train shape (2020, 24, 1)
y_train shape (2020, 24)

8/8 [==============================] - 0s 5ms/step - loss: 6.9319e-06
lstm eval 6.931910775165306e-06
8/8 [==============================] - 0s 4ms/step - loss: 6.1885e-06
lstm eval 6.1885029936092906e-06
8/8 [==============================] - 0s 4ms/step - loss: 6.9583e-06
lstm eval 6.958291578484932e-06
8/8 [==============================] - 0s 5ms/step - loss: 6.0050e-06
lstm eval 6.005032446410041e-06
8/8 [==============================] - 0s 5ms/step - loss: 5.9410e-06
lstm eval 5.9409703681012616e-06
8/8 [==============================] - 0s 6ms/step - loss: 6.2863e-06
lstm eval 6.286334610194899e-06
8/8 [==============================] - 0s 6ms/step - loss: 5.6877e-06 ** lowest loss **
lstm eval 5.6876742746680975e-06
8/8 [==============================] - 0s 8ms/step - loss: 5.9084e-06
lstm eval 5.90835861657979e-06
8/8 [==============================] - 0s 7ms/step - loss: 6.0410e-06
lstm eval 6.041040705895284e-06
8/8 [==============================] - 0s 8ms/step - loss: 6.7053e-06
lstm eval 6.705321084155003e-06
         epochs units skill_np
reg        1000     0  10.2974
rnn 10     5000    10  16.8489
lstm 10    5000    10  13.1366
rnn 20     5000    20  16.6110
lstm 20    5000    20  15.7275
rnn 30     5000    30  16.7144
lstm 30    5000    30  13.0473
rnn 40     5000    40  17.6051  ** highest skill **
lstm 40    5000    40  16.3905
rnn 50     5000    50  15.8601
lstm 50    5000    50  16.6244
rnn 60     5000    60  16.5906
lstm 60    5000    60  15.3780
rnn 70     5000    70  15.9098
lstm 70    5000    70  17.5618
rnn 80     5000    80  16.3284
lstm 80    5000    80  16.7440
rnn 90     5000    90  16.0629
lstm 90    5000    90  16.2596
rnn 100    5000   100  17.4659
lstm 100   5000   100  13.9109
```

- units = [100,200,300,400,500,600,700,800,900,1000] 

```
Batches 2525
Input dimension 24
Forecast horizon 0
Output dimension 24
Units 50
X_train shape (2020, 24, 1)
y_train shape (2020, 24)

8/8 [==============================] - 0s 8ms/step - loss: 5.6642e-06 
lstm eval 5.664193849952426e-06
8/8 [==============================] - 0s 17ms/step - loss: 5.5426e-06 
lstm eval 5.5426135077141225e-06
8/8 [==============================] - 0s 32ms/step - loss: 6.0916e-06 
lstm eval 6.091565410315525e-06
8/8 [==============================] - 0s 50ms/step - loss: 5.6919e-06
lstm eval 5.691922524420079e-06
8/8 [==============================] - 1s 71ms/step - loss: 5.5385e-06
lstm eval 5.538460754905827e-06
8/8 [==============================] - 1s 108ms/step - loss: 5.8707e-06
lstm eval 5.8706609706860036e-06
8/8 [==============================] - 1s 127ms/step - loss: 5.5336e-06
```

- looking at runs from past two days the 36 in performs better

### 10 - lajolla forecast

- import `Load (kW)` from `load_lajolla_processed.csv`
- hourly resample, mean
- rough guess at data input size, units, etc 

```
Batches 174
Input dimension 80
Forecast horizon 0
Output dimension 24
Units 200
X_train shape (139, 80, 1)
y_train shape (139, 24)

1/1 [==============================] - 0s 296us/step - loss: 3.9252e-06
rnn eval 3.925193595932797e-06
1/1 [==============================] - 0s 552us/step - loss: 4.1454e-04
lstm eval 0.0004145356942899525
     epochs units skill_np
reg    1000     0  1.63436
rnn    1000   200  22.9837
lstm   1000   200  14.7794
```

- something different? I forget

### 11 - nwe peak hour forecast

- very simple to turn timeseries into one-hot vector of peaks

```python
df = import_data('actual')
df['peak'] = np.zeros(df.shape[0],dtype=int)

# one-hot vector denoting peaks 
df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:,0], 'peak'] = 1
```

- RNN doens't give a super clean one-hot output (could be helped with ReLU etc), but can take the argmax() of the output vector

- this works pretty well, 80±10 % accuracy with RNN

- LSTM working super well! >90% accurate, maybe even 93-98% (need to do more runs, there is some variability)
  
  ```
  (base) mjw@192 NWE system peak % python sandbox.py
  ```

Batches 2521
Input dimension 24
Forecast horizon 0
Output dimension 24
Units 200
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

1/8 [==>...........................] - ETA: 0s - loss: 8/8 [==============================] - 0s 5ms/step - loss: 0.0173
rnn eval 0.017306644469499588
8/8 [==============================] - 0s 19ms/step - loss: 0.0058
lstm eval 0.005785437300801277
     epochs units  skill_np
rnn    1000   200  0.488095
lstm   1000   200  0.579365
nwe forecast accuracy 0.294
(base) mjw@192 NWE system peak % 

```
-  forecast.py should run with
     - `fcast='peak'` for peak hour forecast, or 
     - `fcast='hourly'` for hourly load forecast

13 - nwe and la jolla peak forecasts

- runtime comparison
  - runtime comparison: matacino, colab, colab GPU, colab TPU
       - RNN and LSTM trained, 24 in 24 out, 200 units, 1000 epochs, 2521 batches

  - matacino: LSTM 6m23s
```

(base) mjw@192 NWE system peak % python forecast.py

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 5ms/step - loss: 0.0230
rnn eval 0.022979749366641045
8/8 [==============================] - 0s 20ms/step - loss: 0.0048
lstm eval 0.004772956483066082

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np         runtime

rnn    1000   200  0.277778 00:02:04.742942
lstm   1000   200  0.607143 00:06:23.831809     

```
- colab normal: LSTM 14m09s
```

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 11ms/step - loss: 0.0180
rnn eval 0.017991919070482254
8/8 [==============================] - 1s 36ms/step - loss: 0.0024
lstm eval 0.0024219935294240713

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np                runtime

rnn    1000   200  0.496032 0 days 00:03:12.567090
lstm   1000   200  0.630952 0 days 00:14:08.724347

```
- colab GPU: LSTM 0m54s, 0m54s
```

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 4ms/step - loss: 0.0093
rnn eval 0.009256313554942608
8/8 [==============================] - 1s 3ms/step - loss: 0.0032
lstm eval 0.003198341466486454

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np                runtime

rnn    1000   200  0.634921 0 days 00:03:09.945725
lstm   1000   200  0.626984 0 days 00:00:53.560267

```

```

Thu May 13 20:18:38 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   34C    P0    27W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
Your runtime has 13.6 gigabytes of available RAM

Mounted at /content/drive

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 - 0s 4ms/step - loss: 0.0181
rnn eval 0.018078191205859184
8/8 - 1s 3ms/step - loss: 9.6180e-04
lstm eval 0.0009618012700229883

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np                runtime

rnn    1000   200  0.488095 0 days 00:03:07.496729
lstm   1000   200  0.654762 0 days 00:00:53.689253

```
- colab TPU: LSTM 0m54s, 17m37s, 15m43s
```

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 4ms/step - loss: 0.0093
rnn eval 0.009256313554942608
8/8 [==============================] - 1s 3ms/step - loss: 0.0032
lstm eval 0.003198341466486454

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np                runtime

rnn    1000   200  0.634921 0 days 00:03:09.945725
lstm   1000   200  0.626984 0 days 00:00:53.560267

```
- added gpu.info
- super long! why?
```

No GPU found
Your runtime has 13.6 gigabytes of available RAM

Mounted at /content/drive

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 11ms/step - loss: 0.0076
rnn eval 0.007647782098501921
8/8 [==============================] - 1s 51ms/step - loss: 0.0037
lstm eval 0.003667251206934452

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np                runtime

rnn    1000   200  0.646825 0 days 00:03:46.990105
lstm   1000   200  0.626984 0 days 00:17:37.635046

```
- again super long?
- ah maybe colab won't allow multiple GPU/TPU sessions
```

No GPU found
Your runtime has 13.6 gigabytes of available RAM

Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 11ms/step - loss: 0.0159
rnn eval 0.01594020612537861
8/8 [==============================] - 1s 44ms/step - loss: 0.0079
lstm eval 0.00788221973925829

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np                runtime

rnn    1000   200  0.507937 0 days 00:03:42.466369
lstm   1000   200   0.56746 0 days 00:16:59.153706

```
- again slow..
```

No GPU found
Your runtime has 13.6 gigabytes of available RAM

Mounted at /content/drive

Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 10ms/step - loss: 0.0228
rnn eval 0.022843575105071068
8/8 [==============================] - 1s 43ms/step - loss: 0.0011
lstm eval 0.0010912209982052445

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units  skill_np                runtime

rnn    1000   200  0.301587 0 days 00:03:32.243483
lstm   1000   200  0.654762 0 days 00:15:43.295834

```
- la jolla load prediction
  - loads and runs fine, rmse_np 12567 kW
  - at 100 units and 1000 epochs, improvement over NP is:
       - linear regression: 28% 
       - RNN: 55% 
       - LSTM: 17%
  - note: showing training loss _after_ epoch 25 is very effective, really shows the slow decrease after initial drop (log-lin or log-log might help also)
```

Site lajolla
Forecast type hourly
Batches 377
Input timesteps 96
Forecast horizon timesteps 0
Output timesteps 96
Units (RNN/LSTM) 100
Epochs 1000
X_train shape (301, 96, 1)
y_train shape (301, 96)

2/2 [==============================] - 0s 5ms/step - loss: 7.6827e-04
rnn eval 0.0007682693540118635
2/2 [==============================] - 0s 8ms/step - loss: 0.0026
lstm eval 0.002555442973971367

np forecast rmse (kW) 12567.3

     epochs units skill_np         runtime

reg    1000     0     0.28 00:00:02.341793
rnn    1000   100     0.55 00:01:07.346767
lstm   1000   100     0.17 00:02:36.050732

```
- nwe peak forecast - grid search on units and epochs (colab GPU)
  - **didn't finish**, but as of search #38 the lowest model loss (evaluation) is 6.1e-7 at search #19 which is 400 units 4000 epochs)

```python
epochs = [1000,2000,3000,4000,5000]
units = [100,200,300,400,500,600,700,800,900,1000] 
```

```
Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 200
Epochs 1000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 1s 3ms/step - loss: 0.0028
lstm eval 0.0027612620033323765
8/8 [==============================] - 1s 3ms/step - loss: 2.9865e-04
lstm eval 0.000298650236800313
8/8 [==============================] - 1s 3ms/step - loss: 4.0442e-06
lstm eval 4.044246907142224e-06
8/8 [==============================] - 1s 3ms/step - loss: 9.0860e-06
lstm eval 9.086029422178399e-06
8/8 [==============================] - 1s 3ms/step - loss: 4.4928e-06
lstm eval 4.492751031648368e-06
8/8 [==============================] - 1s 3ms/step - loss: 0.0025
lstm eval 0.002479780465364456
8/8 [==============================] - 1s 3ms/step - loss: 6.8575e-06
lstm eval 6.8574954639188945e-06
8/8 [==============================] - 1s 3ms/step - loss: 1.1595e-05
lstm eval 1.1594614989007823e-05
8/8 [==============================] - 1s 3ms/step - loss: 2.5550e-06
lstm eval 2.555016635596985e-06
8/8 [==============================] - 1s 4ms/step - loss: 3.8701e-06
lstm eval 3.870053205901058e-06
8/8 [==============================] - 1s 3ms/step - loss: 6.1454e-06
lstm eval 6.1454020396922715e-06
8/8 [==============================] - 1s 3ms/step - loss: 7.8934e-06
lstm eval 7.8934308476164e-06
8/8 [==============================] - 1s 4ms/step - loss: 5.8848e-06
lstm eval 5.884760867047589e-06
8/8 [==============================] - 1s 4ms/step - loss: 2.6727e-06
lstm eval 2.6726743271865416e-06
8/8 [==============================] - 1s 3ms/step - loss: 4.1756e-06
lstm eval 4.1756106838874985e-06
8/8 [==============================] - 1s 4ms/step - loss: 0.0029
lstm eval 0.002929801121354103
8/8 [==============================] - 1s 4ms/step - loss: 6.0059e-06
lstm eval 6.005868272040971e-06
8/8 [==============================] - 1s 4ms/step - loss: 4.1436e-06
lstm eval 4.143556452618213e-06
8/8 [==============================] - 1s 4ms/step - loss: 6.1483e-07
lstm eval 6.148266606942343e-07
8/8 [==============================] - 1s 4ms/step - loss: 7.3050e-07
lstm eval 7.305023927983711e-07
8/8 [==============================] - 1s 4ms/step - loss: 0.0018
lstm eval 0.0018107142532244325
8/8 [==============================] - 1s 5ms/step - loss: 2.2629e-06
lstm eval 2.2629378690908197e-06
8/8 [==============================] - 1s 5ms/step - loss: 3.7992e-04
lstm eval 0.00037991662975400686
8/8 [==============================] - 1s 5ms/step - loss: 1.2694e-06
lstm eval 1.2693606095126597e-06
8/8 [==============================] - 1s 4ms/step - loss: 2.2812e-06
lstm eval 2.281174374729744e-06
8/8 [==============================] - 1s 5ms/step - loss: 4.6216e-06
lstm eval 4.621555945050204e-06
8/8 [==============================] - 1s 5ms/step - loss: 1.7454e-06
lstm eval 1.7453570535508334e-06
8/8 [==============================] - 1s 5ms/step - loss: 1.3500e-06
lstm eval 1.3499798114935402e-06
8/8 [==============================] - 1s 5ms/step - loss: 1.5609e-06
lstm eval 1.560940745548578e-06
8/8 [==============================] - 1s 5ms/step - loss: 7.9627e-04
lstm eval 0.0007962685194797814
8/8 [==============================] - 1s 5ms/step - loss: 0.0034
lstm eval 0.0033992428798228502
8/8 [==============================] - 1s 5ms/step - loss: 7.3211e-06
lstm eval 7.321144494198961e-06
8/8 [==============================] - 1s 5ms/step - loss: 6.4460e-06
lstm eval 6.446015959227225e-06
8/8 [==============================] - 1s 5ms/step - loss: 5.0230e-06
lstm eval 5.0230355554958805e-06
8/8 [==============================] - 1s 5ms/step - loss: 8.8459e-06
lstm eval 8.845928277878556e-06
8/8 [==============================] - 1s 7ms/step - loss: 0.0047
lstm eval 0.00472861947491765
8/8 [==============================] - 1s 6ms/step - loss: 2.2537e-06
lstm eval 2.253667162221973e-06
8/8 [==============================] - 1s 6ms/step - loss: 1.7055e-06
lstm eval 1.7054944692063145e-06
8/8 [==============================] - 1s 6ms/step - loss: 1.0008e-06
lstm eval 1.000754991764552e-06
```

- nwe peak forecast 4000 epochs 400 units
  - lstm is like 99% accurate!

```
Forecast type peak
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) 400
Epochs 4000
X_train shape (2016, 24, 2)
y_train shape (2016, 24)

8/8 [==============================] - 0s 4ms/step - loss: 0.0383
rnn eval 0.038299258798360825
8/8 [==============================] - 1s 4ms/step - loss: 8.8707e-07
lstm eval 8.870654824022495e-07

np forecast accuracy 0.345
nwe forecast accuracy 0.294

     epochs units skill_np                runtime
rnn    4000   400     -0.1 0 days 00:12:58.022295
lstm   4000   400     0.65 0 days 00:03:58.197430
```

### 18 - hyatt grid search

- narrow grid search lstm and rnn, unforunately the results were getting overwritten in the `res` dict so only epochs=3000 shows up
- lowest model.evaluate() loss was 6.9061e-05 from rnn units=300 epochs=2000

```
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Tue May 18 12:33:20 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   48C    P0    36W / 250W |    617MiB / 16280MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
Your runtime has 13.6 gigabytes of available RAM


Site hyatt
Forecast type normal
Batches 528
Input timesteps 96
Forecast horizon timesteps 0
Output timesteps 96
Units (RNN/LSTM) [100, 200, 300]
Epochs [1000, 2000, 3000]
X_train shape (422, 96, 1)
y_train shape (422, 96)

np forecast rmse (kW) 125124.4

         epochs units skill_np                runtime
reg        1000     0     0.14 0 days 00:00:04.746512
rnn 100    3000   100     0.73 0 days 00:14:39.365481
lstm 100   3000   100     0.49 0 days 00:01:18.658814
rnn 200    3000   200     0.79 0 days 00:14:46.743714
lstm 200   3000   200     0.68 0 days 00:01:38.750582
rnn 300    3000   300    -0.14 0 days 00:14:28.154740
lstm 300   3000   300     0.29 0 days 00:02:02.810885
```

- lots of this warning from tf:
  
  ```
  WARNING:tensorflow:6 out of the last 11 calls to <function Model.make_test_function.<locals>.test_function at 0x7faa69275830> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
  ```

### 20 - lajolla with emd

- emd not importing in colab, ughhhhh

- but matacino's conda env `py37keras` has emd installed, runs fine, woot

- Lajolla LSTM u200 e2000
  
  - univariate: 76% skill wrt NP (rmse 21.1 kW on valid set)
  - EMD: 89% skill wrt NP (RMSE 21.1 kW on valid set)

### 29 - lajolla

- whats going on with RMSE error? 

- compare RSME(NP)
  
  - do 24 h NP and calc RMSE on all remaining non-NaN data
    
    - excel 22.310 kW (manually shift NP column)
    - pandas 22.310 kW (`df['np'] = df['Load (kW)'].shift(96)`)
    - awesome
  
  - with pandas do 24 h NP and calc RMSE on test and validate sets
    
    - train: 20.620 kW
    - validate: 20.753 kW
  
  - code (`naive_persistence()`)
    
    - validate: 21.1 kW
  
  - Interesting, so whats going on here is that "batchify_single_series()" takes 
    
    `s = abcdefghijkl` 
  
  - and makes it into 

```
X=   abc       y=   def
     ghi            jkl
```

- so that y does not include all the values of s, just `def` and `jkl` in this example. while this is weird, the results are so good that we shouldn't mess with it!

- also, calculating RMSE on the max-scaled data and then scaling the RMSE back up - seems to work fine.

### Jun

### 1 - lajolla model error

- forecast is REALLY good with certain IMFs included - is it for real?  
  - plot_predictions() RMSE does match up with manual calc (in python, not excel)

### 10 - lajolla response to bad data

- train an LSTM u200 for e2000
- no  bad data
  - model skill 65% (seems low but okay, no IMFs)
- insert bad data: the first sample in the validation set has the 12-hour at zero (4 data points)
  - validation skill for the whole validation data set goes down to 31% !!!
  - seems to big just for one hour of bad data at the beginning of validation set

### 11 - bug found : feeding valid data to training algo

- lstm() (and other models) was being passed X_valid and y_valid instead of X_train and y_train
- model was training on validation data, then validating on validation data
- so.. at least the training was going well (model loss over time), but overfitting and poor generalization

hyatt grid search

- all is not lost: hyatt+IMF3 gets to 7.3% improvement on NP

```
Site hyatt
Forecast type normal
Additional Features IMF3
Batches 528
Input timesteps 96
Forecast horizon timesteps 0
Output timesteps 96
Units (RNN/LSTM) [50, 75, 100, 150, 200, 500]
Epochs [2000]
X_train shape (332, 96, 2)
y_train shape (332, 96)

lstm u50 e2000 eval 0.002183
lstm u75 e2000 eval 0.002295
lstm u100 e2000 eval 0.002394
lstm u150 e2000 eval 0.002267
lstm u200 e2000 eval 0.002259
lstm u500 e2000 eval 0.001462

np forecast rmse (kW) 72.7

    model                             skill_np runtime
    lstm u50 e2000     -0.13         0 days 00:03:13.799136
    lstm u75 e2000     -0.16        0 days 00:04:25.476819
    lstm u100 e2000    -0.19        0 days 00:04:26.544659
    lstm u150 e2000    -0.15        0 days 00:05:25.761023
    lstm u200 e2000    -0.15        0 days 00:05:25.496895
    lstm u500 e2000    0.073        0 days 00:12:26.147912
```

### 12 - nwe normal forecast w/o bug

- again, all is not lost: 59% skill at u100 e500 

```
Site nwe
Forecast type normal
Additional Features 
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0  
Output timesteps 24
X_train shape (1588, 24, 1)
y_train shape (1588, 24)

>>> res, y_valid_pred, hx, = {}, {}, {}
>>> u=100
>>> e=500
>>> name = 'lstm u{} e{}'.format(u,e)
>>> res[name],y_valid_pred[name],hx[name] = lstm(e, X_train, y_train, X_valid, y_valid, n, h, o, u, fcast,Lmax)
lstm u100 e500 eval 0.000022
>>> print_results(res,X_valid,y_valid,y_valid,o,Lmax,fcast)

np forecast rmse (kW) 203.5

                       runtime skill_np
lstm u100 e500 00:10:29.056810     0.59
```

- lstm u50 e2000 gets up to 67% skill

```
Site nwe
Forecast type normal
Additional Features 
Batches 2521
Input timesteps 24
Forecast horizon timesteps 0
Output timesteps 24
Units (RNN/LSTM) [25, 50, 75, 100, 200, 500]
Epochs [2000]
X_train shape (1588, 24, 1)
y_train shape (1588, 24)

lstm u25 e2000 eval 0.000016
lstm u50 e2000 eval 0.000014
lstm u75 e2000 eval 0.000020
lstm u100 e2000 eval 0.000015
lstm u200 e2000 eval 0.000017
lstm u500 e2000 eval 0.000021

np forecast rmse (kW) 203.5

                skill_np                runtime
lstm u25 e2000      0.65 0 days 00:06:28.179119
lstm u50 e2000      0.67 0 days 00:06:23.267284
lstm u75 e2000      0.61 0 days 00:06:27.841748
lstm u100 e2000     0.66 0 days 00:06:16.898422
lstm u200 e2000     0.63 0 days 00:07:25.130314
lstm u500 e2000     0.59 0 days 00:16:25.849390
```

### 14 - how good is the NWE inhouse forecast?

- in-house forecast RMSE
  - if you import '2007-7-9':'2021-4-27' of NWE load data and make a single vector, and then shift a copy of the vector +24h into the future, you can easily calculate the "obvious" naive persistence RMSE (ie with no batching)

```python
>>> vL = get_data('nwe',IS_COLAB=False,additional_features='') # load vector
>>> vL2 = vL[24:]
>>> vLnp = vL[:-24]
>>> root_mean_squared_error(vL2,vLnp)
170.06262798164857
```

-     can also do this with the in-house forecast data, and see that the forecast is about 7% better

```python
>>> vF = import_data_nwe('../../../Google Drive/Data/', 'forecast').values.flatten()
>>> root_mean_squared_error(vL,vF)
158.2068539939649
```

-     but once we slice the data into train-valid-test the validation subset errors are more like 203 (np) and 112 (inhouse forecase). frustrating..

```python
>>> root_mean_squared_error(y_valid,y_valid_pred['np'])*Lmax
203.52773
>>> root_mean_squared_error(y_valid,y_valid_pred['inhouse'])*Lmax
111.987854
```

-     (this is not a problem with house RMSE is calcualted on a matrix vs a vector)

```python
>>> root_mean_squared_error(y_valid.flatten(),y_valid_pred['np'].flatten())*Lmax
203.52773
>>> root_mean_squared_error(y_valid.flatten(),y_valid_pred['inhouse'].flatten())*Lmax
111.987854
```

in-house forecast accuracy at detecting the peak

- however the in-house forecast *as presented* doesn't predict peaks better than naive persistence (OHP onehot of peaks), about 6.5% worse on the **entire dataset**

```python
>>> accuracy_of_onehot_vector(vL_OHP,vF_OHP)
0.2986

>>> accuracy_of_onehot_vector(vP[24:],vP[:-24])
0.3624
```

- on just the validation set it's about the same. so probably no point in using the NWE forecast as a feature.

```
np forecast accuracy 0.318

                   ∆acc  minutes
inhouse       -4.85e-02      NaN
```

### 16 - developing NWE peak forecast

- is the peak (one-hot) approach useful? LIKELY
  - one-hotting the data, a grid search on units and epochs gets a couple models that are able to beat NP by ~1%.. not very good (∆acc is wrt accuracy of NP)

```
lstm u25 e500 {'∆acc': 0.011764705882352955, 'minutes': 1.7085282333333334}
lstm u25 e1000 {'∆acc': 0.01470588235294118, 'minutes': 3.263904783333333}
lstm u50 e500 {'∆acc': -0.007352941176470562, 'minutes': 1.6301929666666666}
lstm u50 e1000 {'∆acc': -0.030882352941176472, 'minutes': 3.22265105}
lstm u75 e500 {'∆acc': -0.022058823529411742, 'minutes': 1.6413757333333334}
lstm u75 e1000 {'∆acc': -0.01470588235294118, 'minutes': 3.2525400166666665}
lstm u100 e500 {'∆acc': -0.01470588235294118, 'minutes': 1.6199915}
lstm u100 e1000 {'∆acc': -0.04264705882352937, 'minutes': 3.207078233333333}
lstm u250 e500 {'∆acc': -0.017647058823529405, 'minutes': 3.417045}
lstm u250 e1000 {'∆acc': -0.03529411764705881, 'minutes': 5.422833333333333}
lstm u500 e500 {'∆acc': 0.007352941176470618, 'minutes': 4.41985325}
lstm u500 e1000 {'∆acc': 0.011764705882352955, 'minutes': 8.41953475}
lstm u1000 e500 {'∆acc': 0.0014705882352941124, 'minutes': 8.437384333333334}
```

- interestingly, I train a lstm u25 e500 on the non-onehot data and the peak accuracy is less than NP

```
>>> accuracy_wrt_peaks(y_valid,y_valid_pred['np'])
0.3176
>>> accuracy_wrt_peaks(y_valid,y_valid_pred['lstm u25 e500'])
0.3147
```

### 30 - NWE peak forecast NP (on newly cleaned load data)

- first: gave the 'actual' load data a second clean based on z-score > 3.. found some negative values, zero values, and super high values. also some high-ish values right around +3sigma but these are likely physical.

- been looking at histogram type distributions of the peak hr, broken down by day of week. clearly the weekends are different, but also friday and monday somewhat. saturday and sunday are also not super similar. there fore a 7-day naive persistence might do better than a 1-day.

- this appears to be true, but just barely. 

```python
df = import_dat_nwe(path,'actual') # new cleaned data
df.columns = ['t']
df['t+24'] =  df.t.shift(-24)
df['t+168'] = df.t.shift(-168) # 7 days
df = df.dropna()

n_days = len(df)/24

dm_t =    df.t.values.reshape(n_days,24) # data matrix
dm_t24 =  df['t+24'].values.reshape(n_days,24)
dm_t168 = df['t+168'].values.reshape(n_days,24)

accuracy_of_NON_onehot_wrt_peaks(dm_t24,dm_t) # 0.3645

accuracy_of_NON_onehot_wrt_peaks(dm_t168,dm_t) # 0.3786
```

### Jul

### 23 - normal forecast

- Beartooth project is wrapped up - maybe less important now to forecast peak hour
- Really should re-visit thesis results, see about evaluating or maybe improving that code
- Maybe also need a benchmark one synthetic dataset thats hard but not impossible to forecast well

### Sep

### 5 - back to the forecast

- peak forecast is where its at? can compare to utility data
- why would you want a peak forecast for commercial building? to plan for demand shaving I guess

### 14 - Pedersen tutorial GRU on Deere data

- [Colab notebook](https://colab.research.google.com/drive/1dfAMqbk5N5ZuGyDDEi8Nu287aPtxJiMp?usp=sharing)

- Initial success! Looks like the MSE halves
  
  - What does that mean for RMSE then?

### 16 - LSTM on Hyatt (Pedersen code)

MSE decreases by 57%, RMSE seems to go down by 34%

- Validation ("test") $MSE$'s on scaled data
  - Naive 1 week = 0.00521
  - LSTM 512x1 + day,hr,dow = 0.00225
    - GRU 512x1 + day,hr,dow very similar
- Validation ("test") $\sqrt{MSE}$  on scaled data

But for some reason when you calculate the RMSE outside of training and then scale it using the `y_scaler` the RMSE decreases only from 73 kW to 61 kW

- Ah! Because the 73 kW is naive 1 day
- Okay, 7 day NP on *all* data is 97 kW whereas 1 day NP on *all* data is 75 kW

Trying to use Pedersen code for multiple layer LSTM isn't going well

- Throws error about the loss function: FML

### Oct

### 4-6 - NWE peak forecast

- so temp does correlate ~0.7 in the summer but in winter its more like -0.6, makes sense right?
- collected load curves by day of week and month of year (7*12 total) for a kind of manual classification, not sure if this will be helpful
  - for certain times of year this isn't bad: januray-sunday peaks are like 82% all in the 7pm hour (so if we guess 7pm for subsequent january sundays we should be right about 80% of the time)
  - the average january-sunday curve has the same peak as ~80% of the population, but this isn't always the case

### 8 - nwe peak forecast

- alf suggested a non-sequential model to reduce the number of parameters, maybe something like:

### 11 - NWE non-ML peak forecast *just from historical distributions*

- like above, calculate 12*7 peak load hour PDFs for each day of week, for each month of year
- forecast example: if tomorrow is an october tuesday (it is), lookup the most likely peak hr for tomorrow based on past october tuesdays
- emanuele suggested only using the last five years (so the past 20 october-tuesdays more or less) due to load growth, but maybe we can confirm the load has grown in a way that retains the same peak hr
- 48% accurate compared to 35% naive persistence
  - "Training data" is the entire dataset except for final 10% for validation
  - NP calculated on same validation set

### 16 - load forecast

- goal: clean up results and write this dang paper
- noticed visually that the 24x1 lstm-emd just predicts the ~hourly average
  - use `std(diff(y_true))=213` as the benchmark and then evaluate `std(diff(y_pred))`
  - actually looks like most of the pedersen models just predict the average, at least on this data set.. I'm not getting nearly the varaibility I did with some two layer lstms from the thesis

### 19 - deere load forecast grid search

- "supercleaned v2" dataset
  
  - was noticing some really weird days in the test set, so did a  "v2" supercleaning of both train and test sets
  - not necessarily bad measurements, but definitely hurting both NP and model learning
  - Original cleaned data: NP1wk =640, lstm24x1+imf3,4,5=527
  - "Supercleaned" data: NP1wk = 681, lstm24x1+imf3,4,5=474
  - "Supercleaned v2"data: NP1wk=595, lstm24x1+imf3,4,5=525

- `sequence_length`
  
  - finding: must the be same for `x_batch` and `y_batch` even if model only predicts a single `y` value for a given row of `x`
  
  - for one row of `x` model calculates only one row (value) of `y` (can try this on trained model)
  
  - for training, Pedersen sets `batch_size=256` which I think is how much data is loaded into optimizer at one time
  
  - these dimensions look familiar, but with other lstm's there was never a problem with different input and output sequence lengths (e.g. 672 in and 96 out)
  
  - oh well..
    
    ```python
    # incomplete! just for reference
    
    x_shape = (batch_size, sequence_length, num_x_signals)
    x_batch = np.zeros(shape=x_shape, dtype=np.float16)
    
    y_shape = (batch_size, sequence_length, num_y_signals)
    y_batch = np.zeros(shape=y_shape, dtype=np.float16)
    
    for i in range(batch_size):
        # This random index points somewhere into the training-data.
        idx = np.random.randint(num_train - sequence_length)
    
        # y is the load shifted ahead of x 1 day
        x_batch[i] = x_train[idx:idx+sequence_length]
        y_batch[i] = y_train[idx:idx+sequence_length] 
    ```
    
    ```python
    >>> print( x_test[:,0:1,:] )
    [[[0.93465733 0.74725275 0.         0.69565217 0.44539105 0.80698092
       0.73180728]]]
    >>> print( model.predict(x_test[:,0:1,:]) )
    [[[0.6704215]]]
    ```

- how `model.predict()` works
  
  - since lstm can calculate a single `y` point from a single observation (right word?) of `x` (see above) its not clear if its better to run `model.predict()` once or a separate time for each day-ahead forecast (like would be done in real life)
  
  - "all at once"
    
    ```python
    # all data at once
    
    y_pred = model.predict(x_test) 
    rmse(y_test, y_pred) # scaled
    # 0.0354
    ```
  
  - "in groups of 96" (batches? sequences? who knows)
    
    ```python
    def predict_in_groups(groupsize=96):
      # scope: knows about model, x_test, and y_test
      begin = 0
      end = groupsize
      err = []
      while end < x_test.shape[1]: 
    
        y_test_1w = y_test[:,begin:end,:]
        x_test_1w = x_test[:,begin:end,:]
    
        y_pred_1w = model.predict(x_test_1w)
        err.append(rmse(y_test_1w, y_pred_1w))
    
        begin = begin + groupsize
        end = end + groupsize
    
      return np.mean(err)
    
    print(predict_in_groups(groupsize=96))
    # 0.0509
    
    print(predict_in_groups(groupsize=672))
    # 0.0359
    ```

- grid search results not so bad!
  
  - lstm 96x2 and 256x2 look good
  - recall that for some reason 3 layer models are not working with Pedersen code

| model          | test_rmse_pred | test_skill : ==bug== | test_std_diff_pred |
| -------------- | -------------- | -------------------- | ------------------ |
| lstm 24x1      | 532.54         |                      | 78.32              |
| lstm 24x2      | 534.99         | 0.11                 | 74.34              |
| lstm 48x1      | 499.27         | 0.19                 | 79.20              |
| lstm 48x2      | 502.76         | 0.18                 | 84.03              |
| lstm 96x1      | 489.44         | 0.22                 | 80.84              |
| **lstm 96x2**  | **451.08**     | **0.32**             | 108.93             |
| lstm 128x1     | 504.38         | 0.18                 | 80.21              |
| lstm 128x2     | 502.29         | 0.18                 | 77.48              |
| lstm 256x1     | 502.48         | 0.18                 | 80.03              |
| **lstm 256x2** | 457.10         | 0.30                 | **115.23**         |
| lstm 512x1     | 477.46         | 0.25                 | 104.62             |
| lstm 512x2     | 472.33         | 0.26                 | 90.53              |

- how does the day-ahead forecast accuracy improve when you increase the data fed to lstm? well..
  
  - here we calculate the rmse (scaled) of the last day of the test set based on giving the lstm just the previous day.. then the previous day plus one hour, then plus two hours.. all the way up to four weeks given to the lstm
    - what we find is that the rmse is minimum for an input sequence of about 230 hrs, asymptotically settling on ~0.0375 for sequences over 400 h)
  - we can also do this not just for the last day, but the second to last day, third, etc
    - find that for the last 7 days of the test set, the one-day forecast rmse (seven of them) all don't decrease much for sequences larger than ~200 hrs
  
  ```python
  # here we predict a day in the test set, with any sequence size going INTO the lstm you want
  def predict_last_day(groupsize=96,end=x_test.shape[1]):
    # scope: knows about model, x_test, and y_test
    begin = end - groupsize
  
    y_test_ss = y_test[:,begin:end,:]
    x_test_ss = x_test[:,begin:end,:]
    y_pred_ss = model.predict(x_test_ss)
  
    return np.mean(rmse(y_test_ss[-96:], y_pred_ss[-96:]))
  
  def someshiz(distance_to_end):
    err = [] # list of rmse's 
  
    n_weeks_in_seq = 4
    for i in range(96,672*n_weeks_in_seq,4):
      err.append(predict_last_day(  groupsize=i,
                                       end=(x_test.shape[1]-distance_to_end)))
    return err
  
  all_err = [] # will be matrix
  
  n_days_to_predict = 7 # consecutive, starting from last day in test set
  for i in range(0,96*n_days_to_predict,96):
    all_err.append(someshiz(i))
  
  all_err_np = np.array(all_err).T # for plotting
  
  t = np.arange(all_err_np.shape[0])
  plt.plot(t,all_err_np)  
  ```

- crank the patience up?
  
  - on the well-performing lstm 96x2 + imf3,4,5 200h sequence
  - loss decreases well at first, asymptotes around 60 epochs, then jumps up and down a few times
  - val_loss is somewhat flat, but does find a minimum around epoch 63, after that up-dn jump in loss
  - not a huge difference from patience=5, but maybe worht doing for larger models especially

- grid: lstm vs lstm-emd?
  
  - `dwh` is day, weekday, hour
  
  - `imf345`is imf's 3, 4, and 5
  
  - model is lstm 96x2, patience=5
  
  - results aren't great, clearly IMFs aren't doing a ton
  
  - but maybe the 96x2 shape is wrong with more IMFs
    
    | lstm 96x2 + ..              | test rmse pred (np=595) | test skill | test std(dff) |
    | --------------------------- | ----------------------- | ---------- | ------------- |
    | load                        | 598.7                   |            | 117.76        |
    | load + day, weekday, hour   | 458.6                   |            | 91.66         |
    | load + dwh + imf3           | 507.4                   |            | 75.3          |
    | load + dwh + imf4           | 478.9                   |            | 84.6          |
    | load + dwh + imf5           | 513.2                   |            | 87.7          |
    | load + dwh + imf3,4,5       | 462.4                   |            | 102.3         |
    | load + dwh + imf3,4,5,10,11 | 572                     |            | 86            |
    | load + dwh + all imfs       | 665                     |            | 62            |

- grid: sequence length *and* units, layers (note the extra IMFs 1 and 2)
  
  - funny how it seems different grid searches give different results even if the same hyperparams are in each search
  - 96 sequence looking stronger than expected given what I can show about forecast accuracy and sequence 
  - next closest to 96 is 1344 (2 weeks) on 256x2 model
  
  ```python
  def run_the_joules( units=24, layers=1, epochs=100,
                      data_points_per_day=96, sequence_length=96*7):
  
    # some meta
    dir = '/content/drive/MyDrive/Colab Notebooks/Models' 
    y, m, d = datetime.now().year-2000, datetime.now().month, datetime.now().day
    path_checkpoint = f'{dir}/lstm {units}x{layers} {y}{m}{d}.keras'
  
    # data
    df = ft.get_dat(  IS_COLAB=True, 
                      site='deere-supercleaned', 
                      features = [  'Load v2 (kW)',
                                    'Day',
                                    'Weekday',
                                    'Hour',
                                    'IMF1',
                                    'IMF2',                                                                 
                                    'IMF3',
                                    'IMF4',
                                    'IMF5',])
    # and so on 
    # ..
  
  # define grid    
  results = {}
  for units in [24,48,96,128,256,512]:
      for seq_len in [96,7*96,14*96]:
        model_name = f'lstm {units}x2 sl{seq_len}'
        results[model_name] = run_the_joules( units=units, 
                                              layers=2,
                                              sequence_length = seq_len)
  ```

```
|                    | test_rmse_pred | test_skill ==bug== | test_std_diff_pred |
| ------------------ | -------------- | ------------------ | ------------------ |
| **lstm 24x2 sl96** | **465.77**     | **0.28**           | 112.29             |
| lstm 24x2 sl672    | 531.00         | 0.12               | 83.48              |
| lstm 24x2 sl1344   | 563.79         | 0.06               | 80.04              |
| lstm 48x2 sl96     | 494.83         | 0.20               | 111.82             |
| lstm 48x2 sl672    | 516.88         | 0.15               | 85.87              |
| lstm 48x2 sl1344   | 569.50         | 0.04               | 82.18              |
| lstm 96x2 sl96     | 733.86         | -0.19              | 92.41              |
| lstm 96x2 sl672    | 641.65         | -0.07              | 77.20              |
| lstm 96x2 sl1344   | 552.68         | 0.08               | 83.05              |
| lstm 128x2 sl96    | 480.54         | **0.24**           | 104.58             |
| lstm 128x2 sl672   | 502.42         | 0.18               | 107.62             |
| lstm 128x2 sl1344  | 634.91         | -0.06              | 86.39              |
| lstm 256x2 sl96    | 492.10         | 0.21               | 138.39             |
| lstm 256x2 sl672   | 543.26         | 0.10               | 88.27              |
| lstm 256x2 sl1344  | 480.44         | **0.24**           | 137.38             |
| lstm 512x2 sl96    | 472.37         | **0.26**           | 134.48             |
| lstm 512x2 sl672   | 531.34         | 0.12               | 100.48             |
| lstm 512x2 sl1344  | 499.29         | 0.19               | 122.13             |

### 19 - hyatt load forecast Pedersen code

- grid search 
- skill not bad but a bit surprising for the smallest, single layer model
- non-linearity doesn't look great (although always tough)
- larger models could maybe use some dropout (loss continues to decrease but val_loss slowly climbs)

```python
# not exact, but gives you an idea of the grid 
for units in [24,48,96,128,256,512]:
for layers in [1,2]:
  run_the_joules(        units=units, layers=layers,
                      site='hyatt',
                      data_points_per_day = 96,
                      sequence_length = 96*7,
                      np_horizon = 96,
                      epochs=100,
                      patience=5,
                      verbose=0,
                      output = True,
                      plots = False,  
                      features = [  'Load (kW)',
                                    'Day',
                                    'Weekday',
                                    'Hour',
                                    'IMF3',
                                    'IMF4',
                                    'IMF5',])    
```

| model         | test_rmse_pred | test_skill ==bug== | test_std_diff_pred |
| ------------- | -------------- | ------------------ | ------------------ |
| **lstm 24x1** | **60.30**      | 17%                | 18.42              |
| lstm 24x2     | 61.93          |                    | 17.75              |
| **lstm 48x1** | 63.83          |                    | **19.06**          |
| lstm 48x2     | 62.96          |                    | 17.80              |
| lstm 96x1     | 62.05          |                    | 18.18              |
| lstm 96x2     | 63.36          |                    | 17.98              |
| lstm 128x1    | 60.87          |                    | 17.74              |
| lstm 128x2    | 61.86          |                    | 18.18              |
| lstm 256x1    | 61.85          |                    | 18.62              |
| lstm 256x2    | 68.00          |                    | 18.99              |
| lstm 512x1    | 63.27          |                    | 18.94              |
| lstm 512x2    | 68.98          | 6%                 | 19.05              |

- grid search with seq_len=96
  
  - overall this helped, but nothing spectacularly better
  - also did seq_len=384 which is a statistical tie with 96
  
  ```python
  # just for reference, not real
  def run_the_joules( 
                      site='hyatt',
                      units=24,
                      layers=1,
                      data_points_per_day = 96,
                      sequence_length = 96*7,
                      np_horizon = 96,
                      epochs=100,
                      patience=5,
                      verbose=0,
                      output = True,
                      plots = False,  
                      features = [  'Load (kW)',
                                    'Day',
                                    'Weekday',
                                    'Hour',
                                    'IMF3',
                                    'IMF4',
                                    'IMF5',],
  
  results = {}
  for units in [24,48,96,128,256,512]:
    for layers in [1,2]:
      results[f'lstm {units}x{layers}'] = run_the_joules( site='hyatt',
                                                          units=units, 
                                                          layers=layers,
                                                          sequence_length=96)        
  ```

|                | test_rmse_pred | test skill | test_std_diff_pred |
| -------------- | -------------- | ---------- | ------------------ |
| lstm 24x1      | 61.31          |            | 20.38              |
| lstm 24x2      | 61.00          |            | 20.68              |
| **lstm 48x1**  | **60.57**      |            | 19.86              |
| **lstm 48x2**  | **60.17**      | 18%        | 19.81              |
| lstm 96x1      | 64.62          |            | 19.79              |
| lstm 96x2      | 61.36          |            | 20.08              |
| **lstm 128x1** | **60.56**      |            | 20.02              |
| lstm 128x2     | 61.39          |            | 20.08              |
| lstm 256x1     | 60.78          |            | 20.55              |
| lstm 256x2     | 61.26          |            | 20.61              |
| **lstm 512x1** | 61.15          |            | **22.07**          |
| lstm 512x2     | 60.89          |            | 20.14              |

### 20 - lajolla, nwe, and terna grid results

- lajolla
  - np 1d test = 33.90 rmse
  - lstm 256x2 = 30.93 rmse
  - lstm 24x1 + imf3,4,5 = 28.85 rmse
  - lstm 128x1 + d,m,wkday = 25.25 rmse
  - lstm 512x2 + d,m,wkday + imf3,4,5 = 25.37 rmse

| units x layer + dropout | test_rmse_pred | test skill (no bug) | test_std_diff_pred | epochs |
| ----------------------- | -------------- | ------------------- | ------------------ | ------ |
| lstm 24x1 do0.0         | 25.55          |                     | 2.83               | 32.0   |
| lstm 24x1 do0.1         | 25.24          |                     | 3.49               | 35.0   |
| lstm 24x2 do0.0         | 24.56          |                     | 2.99               | 18.0   |
| lstm 24x2 do0.1         | 25.17          |                     | 2.95               | 43.0   |
| lstm 48x1 do0.0         | 25.40          |                     | 2.77               | 8.0    |
| lstm 48x1 do0.1         | 24.55          |                     | 3.41               | 30.0   |
| lstm 48x2 do0.0         | 26.86          |                     | 3.10               | 9.0    |
| lstm 48x2 do0.1         | 27.56          |                     | 3.03               | 23.0   |
| lstm 96x1 do0.0         | 25.65          |                     | 3.02               | 14.0   |
| lstm 96x1 do0.1         | 26.76          |                     | 3.13               | 16.0   |
| lstm 96x2 do0.0         | 29.77          |                     | 3.28               | 9.0    |
| lstm 96x2 do0.1         | 26.52          |                     | 2.82               | 13.0   |
| lstm 128x1 do0.0        | 27.00          |                     | 3.24               | 13.0   |
| lstm 128x1 do0.1        | 27.59          |                     | 3.18               | 10.0   |
| lstm 128x2 do0.0        | 27.90          |                     | 3.19               | 9.0    |
| lstm 128x2 do0.1        | 27.30          |                     | 2.88               | 12.0   |
| lstm 256x1 do0.0        | 25.55          |                     | 3.13               | 8.0    |
| lstm 256x1 do0.1        | 25.21          |                     | 3.36               | 9.0    |
| lstm 256x2 do0.0        | 28.56          |                     | 3.54               | 11.0   |
| lstm 256x2 do0.1        | 27.05          |                     | 3.24               | 14.0   |
| lstm 512x1 do0.0        | 25.03          |                     | 3.59               | 9.0    |
| lstm 512x1 do0.1        | 27.04          |                     | 3.90               | 19.0   |
| **lstm 512x2 do0.0**    | **24.34**      | **28%**             | 3.88               | 10.0   |
| lstm 512x2 do0.1        | 24.59          |                     | 3.88               | 11.0   |

- deere "supercleaned v2"
  - np 7d 595 rmse
  - 

### 20 - bug found : skill miscalc

```python
# wrong! skill ranges from +inf to -1
results['test_skill'] = test_rmse_np / results['test_rmse_pred'] - 1

# correct: skill ranges from 1 to -inf
results['test_skill'] = 1 - results['test_rmse_pred'] / test_rmse_np
```

- affects all colab notebooks from 10/19 and 10/20
- impact: 
  - skill old 20% --> skill new ~18%
  - skill old 30% --> skill new ~25%
  - skill old 40% --> skill new ~30%
  - skill old 50% --> skill new ~33%
- probably don't need to re-run all the notebooks, instead just calc the real skill from RMSE

### 21 -  alf paper feedback

- [x] thermal/hydro --> traditional

- [x] **expand on idea of load-following-generation paradigm shift**
  
  - when is it economic?

- [x] medium term forecast --> make clear

- [x] data proccing
  
  - expand on idea that *something* must be constant in model to predict the future

- [x] don't assume 24h correlation is most common

- [x] bi-modal distribution can result in mean value that is not in the domain

- [x] $p$-value not consistent in paper

- [x] Hotel Building 1 dataset referenced before its used

- [x] kW can't be italics

- [x] need to cite LSTM cell 
  
  - "TDS illustrated guide to lstms and grus"

- [x] eqnaray can't have \\\ on last equation

- [x] matrix notation for LSTM equations? why not

- [x] put "generalization error" in quotes and cite Geron

- [x] abstract no acronyms

- [x] acronyms (package glossaries)

- [ ] maybe more detail on the practical process of building the model: shape of input and output data, exactly which hyperparameters and how to tune them, etc

- [ ] could put an image of training data

### Nov

### 8 - lstm load paper v1

- v1 = v0 draft plus alf feedback and my subsequent updates

- sonia feedback on v1
  
  1. the introduction needs to be revised. Sonia suggests: please, do not put subsections. Most important: clearly state the main objectives of the paper in comparison to the existing literature. She doesn't understand what has been done compared to other authors and which gap we want to tackle. And she asks to clarify wether the presented method is innovative or not. 
  
  2. In the formulas, perhaps there is a bit of confusion in terms of letters (used several times), uppercase and lowercase. Some metrics are introduced, but then in Table II slightly different names are used.
  
  3. Some figures are not mentioned in the text.
  
  4. Results: Table II should be better explained: there is a benchmark case (with nothing), then are there results obtained by applying the proposed method? Table II, which is the heart of the results, should be highlighted and it needs to be way more explained, both in its own caption and within the text.

### 18 - weekday vs weekend load forecast

Deere

- All-data (test)
  - NP 7day rmse: 595 kW
  - LSTM-EMD 24x1 rmse: 481 kW
- Weekday (test)
  - NP 5day rmse: 658 kW
  - LSTM 24x1 rmse: 769 kW
- Weekend (test)
  - NP 2 day rmse: 393 kW
  - LSTM 24x1 rmse: 601 kW

### 30 - paper v4

things to add in v4

- computation efficiency burden/paragraph
  - add math for total number of parameters
- EV results - good!
  - ~70% 
- weekend vs weekday forecast on industrial building

## 2022

### Apr

### 20 - redcloud control simulation

- `dogecoin-forecast` works with rnn, gruis

Converting from egauge time to excel time

- egauge: 16434... (10 digits, seconds)

- excel: 44... (5 digits followed by dec, days)

- egauge --> excel
  
  ```python
  excel = (egauge + 2209140000) / 24 / 3600
  ```

### 22 - load forecast

horizon

- previously "horizon" was used incorrectly to mean how far between the most recent measurement (say t=0) and the first forecated point (say t+24h)
- apparently it really means the distance from most recent measurement to the most temporally distance forecasted point
- example
  - model input is t-7d to t=0
  - model output is t+24h to t+72h
  - then the horizon is 72 h, not 24 h

### May

### 23 - Second data cleaning (for OHM)

- Huge + and - outliers, some must be technically impossible (many thousands of kW)
- Pre-cleaning the standard deviation at different resample frequencies (always `mean()`) are quite different.. so the outliers throw off the statistics
```
{'1S': 275.5841788078903, 
 '10S': 99.24827456911919, 
 '1T': 43.31580025634189, 
 '15T': 20.734982794582244}
```
- After cleaning the technically impossible outliers (based on max input and output current from the site) the stdev is more regular
```
{'1S': 19.94984964599502, 
 '10S': 19.511015105009623, 
 '1T': 19.147995803563656, 
 '15T': 18.865029868604722}
```

### Jun

### 9 - Load profile thoughts

- Key issue is still typical year vs actual 2022 measurements etc
- Priority is to estimate how big a difference this can be - especially peaks
- Probably should be normalizing according to temperature
- But then again maybe we need to size for the maximum load over 20 years
- How does this really affect the design? Don't we just overbuild in most situations like this?

### 17 - LSTM paper comments

#### To do

- [ ] Highlight novelty
    - [ ] Fix "novelty" references
- [x] More references
- [ ] Explain negative SS
- [ ] "Benchmark" problem, compare to other forecasts
- [ ] Train/test ratio
- [ ] Is there any result supporting your analysis in subsection IV-B Forecast? There is no result showing the changes of the losses during the training process.
- [ ] Data source?

#### Emanuele/Sonia

I did not receive the mail from *IEEE Transactions on SM*, and it seems that a couple of Reviewers were for "major revision", and the others instead rejected our paper even if it is not clearly stated.  However they pointed out that:

1. the novelty of the paper should be highlighted/explained more.  Apparently they do not see any difference with recent publications on LSTM + EMD for Load Forecast

2. more benchmark models should be added (not only Naive Persistance) for a better comparison

3. bibliographic review should be increased with more recent papers  on the same topic emphasizing the differences between or paper and the  state-of-the-art.

4. Besides, some explanations should be added i.e. R1 negative Skill Score  in Table II and (maybe) we can also add a link to the employed datasets.

..

Now, we can submit the paper "as is" to a different scientific journal  as "Energy Conversion and Management" or change it a little according to the Reviewers' suggestion and resubmit it to another IEEE Transactions  as "IEEE Transactions on Power Systems" (we will decide next week together with Sonia, who is reading in copy).

#### IEEE

Editor's Comments:

**Editor**
Comments to the Author:
The reviewers have raised questions regarding the technical issues of the proposed method, novelty and contribution of the paper, among others. In particular:

1. The paper lacks scientific novelty. The authors have not presented a strong case justifying novelty or demonstrating sufficient contributions to the new body of knowledge.
2. Literature survey is not sufficient to present the most updated R&D status for further justification of the originality of the manuscript.
3. The details of the experiments are not clear, for example it is not evident how sensitive is the performance to various parameters.
4. *The work presented in the paper has not been thoroughly benchmarked. The authors should benchmark their work with recently published works for performance validation.*
5. The authors have not provided insightful analysis and thorough comparison of results.

Reviewers' Comments:

**Reviewer: 1**

Comments to the Author
This paper proposed a comprehensive LSTM-EMD methodology for dayahead building load forecast.
This paper is well-written and well-organized.
This reviewer has following comments.

1. The contribution of this paper should be highlighted. Is it the improvement in the areas of data pre-processing and feature engineering? Or the implementation in various case studies?
2. As the authors pointed out in section IV. B, it is difficult to compare with other works considering the lack of same benchmark. Thus, it would be great if the authors can comment on why benchmark naive persistence model can be a acceptable benchmark while representing state-of-the-art implementations.
3. It would be great if the authors can commnet on the -1% skill socre in row 3a in Table II. It is understandable that adding IMF increase SS. However, how often will the negative SS happen in other case studies? Will IMF always ameliorate this problem?

**Reviewer: 2**

Comments to the Author
This paper used a LSTM-EMD methodology to get building-scale load forecasting results, and several datasets are used. Overall, the novelty of this paper is weak and the language also needs to be improved. The reviewer has the following comments.
1. It seems that the normal LSTM and EMD methods are used in building-scale load forecasting. What are the differences between this paper and other literature? The innovation of this paper is totally not clear.
2. What is the ratio of the training set to test set?
3. Is there any result supporting your analysis in subsection IV-B Forecast? There is no result showing the changes of the losses during the training process.
4. The benchmark only includes the naive persistence model, which is not enough to show the superiority of the proposed method. A clear comparison to the-state-of-the-art methods is required.

**Reviewer: 3**

Comments to the Author
The paper proposes an LSTM-EMD based day-ahead demand forecasting method for buildings. The topic is interesting and the paper is nice to read. However, the reviewer has major concern on the original contribution of the proposed work in the paper with respect to the existing literature.

For a data-driven method, what is the fundamental difference between the demand forecasting problem for buildings than that for transmission/distribution systems which have been proposed and analyzed by quite a few existing studies?

Meanwhile, LSTM has been proposed and widely used in demand forecasting. EMD has also been applied in the electricity demand forecasting problem, e.g., [A]. The authors are suggested to further highlight the original contribution and technical innovation of the proposed work in the paper.
[A] J. Bedi and D. Toshniwal, "Empirical Mode Decomposition Based Deep Learning for Electricity Demand Forecasting," in IEEE Access, vol. 6, pp. 49144-49156, 2018.

The authors are suggested to introduce the sources of the datasets in the case study more explicitly in the paper.

More comparative analysis with other existing forecasting methods in the case study is welcome.

**Reviewer: 4**

Comments to the Author
The proposed day-ahead building load forecasting model is just a combination of previously published approaches, including the vanilla LSTM model, Empirical Mode Decomposition for feature engineering, and Hyperband for hyperparameter tuning. In particular, there have been numerous studies that combine EMD with short-term load forecasting, hence this paper cannot be regarded innovative. Furthermore, the benchmark only includes a naive persistence model; other competing models and metrics should be examined.
Despite comprehensive data analysis, including autocorrelation, normality, stationarity, and outliers, the research lacks sufficient innovation to be published on IEEE TSG. 

### Aug

### 17 - Ashley intern review

Power outage detection

- Python function to find outages based on:
    - Values below a threshold
    - Fast drops regardless of threshold

Predicting solar production

- DHI and GHI correlate somewhat

Convert Survivability Chart (R $\rarr$ python)

- Check

Mining simulator

- Python to simulate mining on BR WWTP excess solar (after battery full)
- Get BTC price from Yahoo finance library

1h vs 1min data

- WWTP and HWC Data from Entech
- She caught a 5 h time shift

Aspen Yearly Overlay

- Check

REopt / SQlite Table

- Python code to build load profiles from SQlite table

Peak prediction

- See where PRPA day-ahead forecast is above measured peak of the month: call that the peak period where we would want to dispatch battery
- If she predicts the hours 18, 19, 20 then she rearranges the hours according to biggest difference between forecast and real monthly peak (ie could be 19, 18, 20)
- Current FoCo accuracy for predicting peak hour of the month is about 50% (she does somewhat better)
- However in summer ashleys algo dispatches the battery more often (say 25 h out of the month), which could be fine for a battery but not great for load shedding (if rate payer needs the load)

Demand response e-mail ➡️python

- Zapier to python



### Oct

- When your password expires task scheduler stops running (or maybe stops having admin rights?)
- Closing session vs logout
    - He just closes session, says its doesn't matter which you do
- Server never sleeps



### 27 - Controller 

Travis slack

> Here is the current format for the input data to Redcloud Real Time. Timesteps are numbered as 1,2,3...x where 1 is the first timestep coming up, 2 is the second, and so on. X is variable, could be 96, 192, etc. So if you run RRT at 8:05 am, timestep 1 could represent 815am. If you run it at 8:20am, timestep 1 could be 830am. TimeDx (1-4 shown) are the timesteps that are in Demand ratchet x.
>
> Loade is the upcoming electric load. The specification is timestep1 load1 timestep 2 load2
>
> (Currently, I assume that the solar is netted against the load in preprocessing. So RRT doesn't actually deal with solar right now for simplicity.)
>
> rate is the TOU rate, i.e. cost per kWh of electricity at each timestep. Specified as timestep1 rate 1 and so on.
>
> Demandrates are the demand rate for each demand window. Specified as ratewindow1 rate1, where ratewindow is the timed1 noted above. So basically rate1 is the demand rate for the timesteps in TimeD1 and so son.
>
> **Inputs.dat:**
>
> ```vbscript
> set timed1 := 1 2 3 5 ; 
> set timed2 := 4;
> set timed3 := ;
> set timed4 := ;
> param loade := 1 10 2 20 3 36 4 38 5 30 ;
> param rate := 1 .1 2 .1 3 .05 4 .1 5 .1 ;
> param demandrates := 1 10 2 20;
> ```

- So for `loade`, `rate`, and `demandrates` the odd values are indices and the even values are data

### 28 - Solar forecast

Jail solar naïve persistence MAE

- 1 day 6.8 kW (7.0% of max)
- 2 day 7.6 kW (7.8% of max)
- 7 day 7.6 kW (7.8% of max)

### Nov

### 3 - pyomo and load-forecast

*lots of doco in repo: readme, python notebook etc*

pyomo

- pyomo works! needed to add gkpl (or whatever) to windows PATH

load-forecast

- load forecast obviously need to load and run model (previously trained elsewhere)

- this wasn't working, though I found some sample code that runs fine with environment `tf-pyo`

    ```shell
    python==3.6
    keras==2.9.0
    tensorflow==2.9.2
    pyyaml==6.0
    h5py==3.1.0
    pyomo==6.4.2
    pvlib==0.9.3
    ipykernel
    pandas
    plotly              # optional
    nbformat>=4.2.0     # for plotly
    ```



### 8 - controller / mugrid dev call

*Travis: interested in talking about what we're doing in the field*

Longmont / Veloce

- "Just freed up $60k by moving to Veloce"
- Travis more and more interested in having our own, on-site box, thinks it will be mandatory for some BESS vendors
- Lets assume we need to talk to their Vport over modbus

### 9 - dev call

- Looking for a demo
    - PRPA scraping
    - LSTM load forecast ("people love ML")
    - Redcloud Realtime plots

## 2023

### Mar

### 27 - Redcloud realtime

Real projects in order

1. Courthouse jail
2. Longmont
3. Verizon
4. Highway garage (similar to courthouse jail)
5. Stevens.. ?

Demo 2.0

- KPI: what is the utility bill in this situation



## 2024

### Jun

#### 9

- fundamental problem with how validation is done
  - `x_train` has the shape `(batch_size, n_input_measurements, n_x_signals)`
  - `y_train` has the shape `(batch_size, n_output_measurements, n_y_signals)`
- but!
  - `x_valid` has the shape `(1, some_large_number, n_x_signals)`
  - `y_valid` has the shape `(1, some, large, number, n_y_signals)`
  - ie the validation data is always a single sequence passed into LSTM, bizzare since normally we limit the input to `n_input_measurements`

#### 12

- `////////// units_layers=[96, 24] dropout=[0.1, 0.1] n_in=12 loss=mse//////////`
- vloss 0.0042

### Oct

#### 16 - Code v1.4 vs v1.5 results

|                       | v1.4_persist1D                                     | v1.5                                                         |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| Code                  |                                                    | Some changes, doesn't seem substantial                       |
| Benchmark             | Persist calc'd 1d                                  | Persist calc'd 7d                                            |
| Search Space Features | 2 (load, persist1workday)<br />4 (" ", IMF3, IMF4) | 1 (load)<br />2 (" ", persist1workday)<br />4 (" ", IMF3, IMF4) |
| Completion            | 150 models                                         | 33 models                                                    |
| Best model            | u12-96_d0.1_in96_flen4                             | u4-24_d0.1_in288_flen1                                       |
| Best skill            | 29.8%                                              | 8.1%                                                         |
| Best % days pos skill |                                                    |                                                              |
| Note                  |                                                    | How did this even do random search? That code is commented out |

# Meetings

## 2021

### Mar 3 - PhD Kickoff

*Emanuele, Sonia, Travis, Amy*

- Good opportunity: battery control
- Possible opportunity: load control
- NREL paper on how much load profiles even matter
  - Building load profiles maybe designed for annual usage unlike a TMY file which tries to not be the average
  - Also building profiles may use hourly interval
- Some exams: stats, python (also data camp)
- Lorenzo someone?
- Simone polimeni last year phd student
- Monthly standing meeting agreed

### Apr 7 - Standing

- T: hawaii is pretty consistent weather and daily behavior, San Diego (La Jolla) much less so, may need to use Cooling Degree Days to predict load
- E: leave covid out of the the La Jolla data
- A: we used a truth file vs an expected variability file in aircraft testing
- Planning requires a typical load profile, operation/control needs to contain all the forecast uncertainty
- Goals

1. E: Finish 24 h LSTM forecast
2. M: Consider reframing forecast problem for La Jolla demand target

### Apr 9 - Emanuele & Alfredo

- Data normality/stationarity

  - Shapiro wilk root test (ADF-like) for normality
    - A: When you have a huge amount of data these tests tend to fail

- Features

  - Separate out IMF 3 and 4?
    - A: easy for models to put features together, hard for them to pull them apart, downside is of course a larger model    
  - Exogenous variables

- Testing

  - Cross validation

- Bitcoin mining

  - Alfredo has a friend

### Jul 23 - Emanuele

- Paper
  - Load forecast for LaJolla and Hyatt
  - Peak hour forecast for NWE (maybe including magnitude)
- Exogenous Features
  - ==Temperature (forecast, forecast mean)==
  - Intraday market prices
- Models
  - =="Smart persistence models"==

### Jul 26 - Emanuele mamba demonstration

- "Now I get it"
- ==See if there is a bias in the error of LSTM non-peak forecast at predicting peaks==
- Dispatch diagram.. ?
- Resilience course: https://www.corsoram-phm.energia.polimi.it/

### Oct 8 - Emanuele paper review

1. Peak hour classification using temp, energy, holiday, day of week, other things as features 
   1. Use ANN
2. Check the daily error on the peak magnitude using current load forecast
3. Symmetrical MAPE



- Lit review in intro
- Explain references rather than just a bunch of numbers
- Intro
  - P1 - set problem
  - P_last - short summary of sections of paper
- Methodology flow chart more simple, include inputs and outputs, put the text of each box in the body
- "Normality" open sentence introduces the idea but doesn't really conclude anything
- K-means coupling is quite common, don't need the pseudocode (but do want the math because its more rigorous)

### Oct 12 - Lab lunch meeting

- Platoon / Musetta

  - IT project
  - Pull in all data from microgrid to aws for analytics and management

- Prophet

  - EV forecast / Alf
    - Dont know power curve during charging period, just beginning and ending time and total energy
    - Two other datasets in addition to UCSD
    - Luca working on using traffic data (e.g. NYC) to see if, hypothetically 10% of the cars were electric, you could give drivers a kind of signal for when and where they should charge based on the likelihood of there being chargers and power available

- Cell SOC characterization / Panos

  - Lots of fun things

- Nowcasting / Ale

  - All sky camera
  - Using satellite data (too?)
  - Clearness index, five classes

### Oct 14 - Monthly mugrid-polimi call

- microgrid lab
  - starting from new year could start to do peak shaving tests
  - in the meantime can define the setup
- my update (see slides)
  - possible collaborators (one does more forecasting, the other more ML)
    - [Emanuele della Valle](https://www.deib.polimi.it/eng/people/details/170608)
    - [Marco Brambilla](https://marco-brambilla.com/) - data science
- travis
  - we have these real projects coming up, need to know whats possible
  - interested to see how statistical analysis of peaks can be put into practice

## 2022

### Jan 5 - Dev planning

Dev planning 2022

- Bayfield peak shaving

  - How to formulate and solve MILP

    > Travis already has super basic port of redcloud to pyomo, Mike to continue
    > Don't bang head against wall here (also the load and solar data is the hard part)
    > Big ones: MOSL (server), Xpress (server), GAMS, AMPLE

  - Implementation questions

    - Talking to ELM

      > Will figure out while talking to Switch Storage during Bayfield process

    - Server vs cloud vs local

      > Not the server for sure
      > Issue with cloud machine is monthly cost (use google $300 during dev)
      > Issue with local machine is dev cost and reliability

    - Usefulness of Prescient?

      > TBD

- Longmont dispatch

  > Haven't heard back from them in a long time (but not worried)

- Solar forecasting

  - Day ahead vs month ahead

  - Variability and justification for expected demand savings (pure solar or battery)

- Load forecasting

  - Day ahead vs month ahead vs long term

  - Peak vs normal

  - Load study

    > What peaks are we missing in 15-min or 60-min profiles?

- Data harvesting

  > Double servers?
  > New project with Shrujan (sp?) for 1-second data

- EV

- Mining

- Tools

### Jan 13 - Discuss paper v4

- highlight strenghts in title
- IEEE page limit??
- paper novelties
  - EMD
  - comprehensive methodology leading to consistent results across case studies
  - pick IMFs w/ cross correlation
- methodology flow diagram
  - consider 
- biopics

### Jan 17 - Intro with emanuele della valle (DEIB)

- Can offer: data, my time, grant writing
- Alessio and Giacomo
  - Working on dataset from RSE on predicting maintenance in Milano dx
- Streaming learning appears to outperform trad batch oriented ML tech
- Computer science so more interested in methods (but we have data so this is good)
- Academic for now, see about more possibilities in future between Quantia/muGrid
- Quantia collaboration with InFlux DB
- RSE
  - Maurizio Delfanti
  - Roberto Infantini
  - Tominelli/Tornelli
  - Bionda
- Next steps
  - Present to Alessio and Giacomo and vice versa

### Jan 20 - Bayfield controller

- Want to have some eyecandy for sales: videos, app, dashboard, not exactly clear
- Audiences:
  - People who have our system (bayfield and longmont)
  - People who have a different system that doesn't work
  - People who dont have a system and should want a mugrid one
  - Potential acquirers of mugrid

### Jan 26 - Collaboration with DEIB

*emmanuele della valle, alessio bernardo, giacomo ziffer*

- Alessio
  - 3rd year phd?
  - Streaming data
  - Works in non-stationarity
  - Presentation
    - Streaming data is unbounded, constantly arriving, changing statistical properties etc
    - Data distribution may change with time
    - Unbalance means you get a mix of samples (say class 1 and class 2) in a proportion that you don't know a priori 
    - Models need to be online, dynamic to handle concept drift
- Giacomo
  - First year phd, but couple year research fellow
  - Similar to alessio: ML models for stream of data (adapt to changes and learn over time)
  - Kalman filter
  - Interested in temporal dependence on how models learn

### Feb 2 - Collab DEIB

- Computationally expensive to constantly retrain, also you may have to hold a lot of data in RAM
  - Also you may have concept drift in between retraining 
- Particularly import in the era of big data
- Offline learning can possibly perform better for a fixed time scenario
- Weekend vs weekday
  - Probably a typical case of concept drift, say Concept 1 and Concept 2
- Concept drift can be a hard moment in time or something more gradual
- *Machine learning for data streams with practical examples* - Albert Bifet (2018)
- Usually we assume data is stationary and independent
- Many streaming machine learning approaches will use electricty prices or consumption
- Gradient-based training assumes data can be perfectly shuffled
  - Doesn't hold when IID assumption isn't true
- Hard to learn multiple tasks at the same time, "catastrophic forgetting"
- Plasticity vs stability tradeoff 
  - Maybe we can learn concept 1 to full performance, but then maybe we dont learn a new concept
- State of the art
  - May use change point (concept drift) detection, but this is an ad-hoc solution, not principled
  - Concept drift -> assume same distribution only during same concept
  - Temporal dependence -> assume short-term correlations

### Feb 16 - Collab DEIB

*Alessio, Giacomo, Emanuele Della Valle*

- Not uncommon that simple model outperforms complex models 
- Kappa statistic for performance
- Online very fast decision tree slightly better than Simple Exponential Smoothing

### Feb 18 - Emanuele

- Re-focus on operational forecast
  - Giulia Magistrati 
- Then go back to load forecast and see if it should be improved
- RE process
  1. load and solar forecast
  2. simulated, realistic dispatch without perfect foresight
  3. optimal asset sizing
  4. online operational dispatch

### Mar 3 - Collab DEIB

- usually get new data and retrain immediately
  - make prediction $y_{t+1}$
  - immediately calculate error and retrain when you get $x_{t+1}$
- "we shifted the training process 96 points into the future.. we call it `evaluate_delayed()` 
- their best model was HTReg (hoefding tree regressor) with all IMFs
  - ~7% increase on "test_skill"
- with my LSTM-EMT (all IMFs) 256x1 they got about 6% SS
- they use river ML package for streaming learning
- to do
  - I need to see if I can reproduce my 18 or 20% results on hyatta
  - or, find a better data set

### Mar 17 - Emanuele

- For testing on microgrid talk to Riccardo Simonetti
- Could use code from Simone Polimeni



### May 5 -  Collab

Changes to my LSTM code

- Added test set

- Avoid the problem of labels being used during training

- Test 70%, valid 20%, test 10%

- LSTM could be more simple, only need the past couple hours of data

- Randomized the batches but didn't change the randomized batch order

- Also noticed that randomizing batches didn't seem to help much

- Best results with all IMFs rather than just a few

- For each configuration (hyperparams) run 10 different models starting with 10 different seed values

  - Take the mean of these for "average SS"

- Best instances at 1/2 day or 1 day

- Best lstm hidden layer size 50

- Best lstm 18% SS

- Best streaming 13%

  - Less memory, training time, etc

Paper strategy

- **If there is no other comparison between SML and batch ML we should publish these results pretty soon**

- The kind of drift they are concerned with is unlabellable

  - Weekends and holidays are easy to label, so do that and make the model learn it

  - Temperature change is related to concept drift but also has an exogenous correlation

  - Covid is a hard to label of course

Forecasting/Regression SML

- Usually work with one sample at a time, also work with microbatches

- But streaming doesn't mean you can't use it for microbatches
