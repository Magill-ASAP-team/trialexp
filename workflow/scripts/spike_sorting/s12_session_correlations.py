'''
Script to perform correlations between spikes and photometry for each event type

'''
#%%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import xarray as xr
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from snakehelper.SnakeIOHelper import getSnake

from trialexp.process.ephys.utils import calculate_pearson_lags
from trialexp.process.group_analysis.plot_utils import style_plot
from trialexp.process.pyphotometry.utils import *
import settings
import itertools
from tqdm.auto import tqdm
from loguru import logger

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/xr_corr.nc'],
  'session_correlations')

#%% Path definitions
verbose = True
root_path = Path(os.environ['SESSION_ROOT_DIR'])

#%% Loading files
xr_spike_trial = xr.open_dataset(sinput.xr_spikes_trials) #file is huge, use lazy loading
session_root_path = Path(sinput.xr_spike_fr).parent
xr_session = xr.load_dataset(session_root_path/'xr_session.nc')
xr_spike_fr_interp = xr_spike_trial.interp(spk_event_time=xr_session.event_time)

#%% Calculate cross-correlation between unit activity and photometry signals
photom_vars = ['_zscored_df_over_f', '_zscored_df_over_f_analog_2']
var = [v.replace(photom_vars[0],'') for v in xr_session.data_vars.keys() if v.endswith(photom_vars[0])]
var2analyze = list(itertools.product(var, photom_vars))

def analyze_correlation(fr_data, photom_data, name, evt_time_step, cluID):
    # Calculate the correlation of each units with the photometry signal
    photom_data = np.squeeze(photom_data)
    
    print(f'Processing {name}')
    max_lags = 50 # corresponds to arround 400ms at 50Hz
    lag_step = 5
    nlags = max_lags//lag_step
    
    corr = np.zeros((fr_data.shape[2],nlags*2+1))
    for i in range(fr_data.shape[2]):
        # Negative lag means the photometry signal will be shifted left
        lags,_, corr[i,:] = calculate_pearson_lags(fr_data[:,:,i], photom_data,max_lags, lag_step)
    
    xr_data = xr.DataArray(corr,
                           name = name,
                           dims=['cluID', 'lag'],
                           coords={'cludID':cluID, 'lag':lags*evt_time_step})
    return xr_data

# Calculate the time step for each unit of lag
evt_time = xr_spike_fr_interp.spk_event_time
evt_time_step = np.mean(np.diff(evt_time))
#TODO do correlation on the average curve instead
results = Parallel(n_jobs=20, verbose=5)(delayed(analyze_correlation)(xr_spike_fr_interp[f'spikes_FR.{evt_name}'].data,
                                                           xr_session[f'{evt_name}{sig_name}'].data,
                                                           evt_name+sig_name,
                                                           evt_time_step,
                                                           xr_spike_trial.cluID) for evt_name, sig_name in var2analyze)

#%% Save
xr_corr = xr.merge(results)
xr_corr.to_netcdf(soutput.xr_corr, engine='h5netcdf')
xr_spike_trial.close()

#%%  Plot some example of units with high correlation

evt_name = 'hold_for_water'
sig_name = '_zscored_df_over_f_analog_2'
var_name = evt_name+sig_name

corr = xr_corr[var_name].data
max_corr = corr.max(axis=1)
lag = xr_corr.lag.data
max_corr_loc = lag[corr.argmax(axis=1)]
sorted_idx = np.argsort(max_corr)[::-1] #sort the coefficient in descending order

photom_data = np.squeeze(xr_session[var_name].values)
fr = xr_spike_fr_interp[f'spikes_FR.{evt_name}'].data
# plt.plot(fr.mean(axis=0))

fig, axes = plt.subplots(3,3,figsize=(3*4,3*3))
label1 = None
label2 = None
for i,ax in enumerate(axes.flat):
    ax2 = ax.twinx()

    if i == len(axes.flat)-1:
        label1='unit firing'
        label2='photmetry'
    
    x = fr[:,:,sorted_idx[i]].mean(axis=0)
    y = np.nanmean(photom_data,axis=0)
    shift = int(max_corr_loc[sorted_idx[i]]/(1000/50))
    c = np.corrcoef(x[:shift],y[-shift:])[0,1]
    ax.plot(x[:shift], label=label1)
    ax2.plot(y[-shift:],'r', label=label2)
    ax.set_title(f'{c:.2f} Max. corr = {max_corr[sorted_idx[i]]:.2f} at {max_corr_loc[sorted_idx[i]]:.1f}ms')

fig.tight_layout()
fig.legend()
# %%
