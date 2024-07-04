#!/usr/bin/env python

import os
import pandas as pd

gs_dir = 'results/bayfield_jail-courthouse_v1.3_rand_max10epochs/'

dirs = next(os.walk(gs_dir))[1]

results = pd.DataFrame({'model':[],'mean_skill':[]})
for d in dirs:
    files = os.listdir(gs_dir+d)
    if 'errors.csv' in files:
        e = pd.read_csv(gs_dir+d+'/errors.csv',index_col=1,parse_dates=True)
        mean_skill = e.skill_mae.mean().round(3)
        results.loc[len(results)] = {'model':d,'mean_skill':mean_skill}

results = results.sort_values(by=['mean_skill'],ascending=False)
results.to_csv(gs_dir+'results_summary.csv')
print(results)
