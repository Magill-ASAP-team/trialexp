'''
Example script to perform correlations between spikes and photometry
for the whole session
#TODO: too slow and not very informative, probably need to focus on task region only 
# disabled in workflow for now
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

# %% Path definitions

verbose = True
root_path = Path(os.environ['SESSION_ROOT_DIR'])

# %% Loading files
xr_spike_trial = xr.open_dataset(sinput.xr_spikes_trials) #file is huge, use lazy loading

session_root_path = Path(sinput.xr_spike_fr).parent

xr_session = xr.load_dataset(session_root_path/'xr_session.nc')

xr_spike_fr_interp = xr_spike_trial.interp(spk_event_time=xr_session.event_time)
# xr_spike_session = xr.merge([xr_session, xr_spike_fr_interp]) # make sure their time coord is the same

df_events_cond = pd.read_pickle(session_root_path/'df_events_cond.pkl')
df_pycontrol = pd.read_pickle(session_root_path/'df_pycontrol.pkl')

#%%
# calculate the maximum cross-correlation within a range
photom_vars = ['_zscored_df_over_f', '_zscored_df_over_f_analog_2']
var = [v.replace(photom_vars[0],'') for v in xr_session.data_vars.keys() if v.endswith(photom_vars[0])]
var2analyze = list(itertools.product(var, photom_vars))

def analyze_correlation(fr_data, photom_data, name, evt_time_step):
    # Calculate the correlation of each units with the photometry signal
    photom_data = np.squeeze(photom_data)
    
    print(f'Processing {name}')
    max_lags = 50
    lag_step = 2
    nlags = max_lags//lag_step
    
    corr = np.zeros((fr_data.shape[2],nlags*2+1))
    for i in range(fr_data.shape[2]):
        lags,_, corr[i,:] = calculate_pearson_lags(fr_data[:,:,i], photom_data,max_lags, lag_step)
    
    xr_data = xr.DataArray(corr,
                           name = name,
                           dims=['cluID', 'lag'],
                           coords={'cludID':fr.cluID, 'lag':lags*evt_time_step})
    return xr_data

evt_time = xr_spike_fr_interp.spk_event_time
evt_time_step = np.mean(np.diff(evt_time))
results = Parallel(n_jobs=20, verbose=5)(delayed(analyze_correlation)(xr_spike_fr_interp[f'spikes_FR.{evt_name}'].data,
                                                           xr_session[f'{evt_name}{sig_name}'].data,
                                                           evt_name+sig_name,
                                                           evt_time_step) for evt_name, sig_name in var2analyze)

#%%
xr_corr = xr.merge(results)
xr_corr.to_netcdf(soutput.xr_corr, engine='h5netcdf')

#%%  Plot some example of units with high correlation


