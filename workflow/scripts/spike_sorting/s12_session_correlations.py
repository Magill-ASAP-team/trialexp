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
import seaborn as sns
from trialexp.process.ephys.photom_correlation import plot_extrem_corr, get_corr_spatial_distribution

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
df_metrics = pd.read_pickle(sinput.df_quality_metrics)


#%% Calculate cross-correlation between unit activity and photometry signals
photom_vars = ['_zscored_df_over_f', '_zscored_df_over_f_analog_2']
var = [v.replace(photom_vars[0],'') for v in xr_session.data_vars.keys() if v.endswith(photom_vars[0])]
var2analyze = list(itertools.product(var, photom_vars))
evt_time = xr_spike_fr_interp.spk_event_time
evt_time_step = np.mean(np.diff(evt_time))

def analyze_correlation(fr_data, photom_data, name, evt_time_step, cluID, average_trial):
    # Calculate the correlation of each units with the photometry signal
    photom_data = np.squeeze(photom_data)
    
    print(f'Processing {name}')
    max_lags = 5 # corresponds to arround 100ms at 50Hz, larger shift may risk mixing with other events
    lag_step = 1
    nlags = max_lags//lag_step
    
    corr = np.zeros((fr_data.shape[2],nlags*2+1))
    for i in range(fr_data.shape[2]):
        # Negative lag means the photometry signal will be shifted left
        lags,_, corr[i,:] = calculate_pearson_lags(fr_data[:,:,i], photom_data,max_lags, lag_step, average_trial)
    
    xr_data = xr.DataArray(corr,
                           name = name,
                           dims=['cluID', 'lag'],
                           coords={'cluID':cluID, 'lag':lags*evt_time_step})
    return xr_data


#%% Calculate the time step for each unit of lag
evt_time = xr_spike_fr_interp.spk_event_time
evt_time_step = np.mean(np.diff(evt_time))
#TODO do correlation on the average curve instead
results = Parallel(n_jobs=20, verbose=5)(delayed(analyze_correlation)(xr_spike_fr_interp[f'spikes_FR.{evt_name}'].data,
                                                           xr_session[f'{evt_name}{sig_name}'].data,
                                                           evt_name+sig_name,
                                                           evt_time_step,
                                                           xr_spike_trial.cluID,
                                                           average_trial=True) for evt_name, sig_name in var2analyze)

#%% Save
xr_corr = xr.merge(results)
xr_corr.to_netcdf(soutput.xr_corr, engine='h5netcdf')
xr_spike_trial.close()


#%% Plot the correlation figures
for evt_name, sig_name in var2analyze:
    fig = plot_extrem_corr(xr_corr, xr_spike_fr_interp, xr_session, evt_name, sig_name)
    fig.savefig(Path(soutput.corr_plots)/f'corr_{evt_name}_{sig_name}.png',dpi=200)

# %% plot the overall distribution
sig_names = ['_zscored_df_over_f','_zscored_df_over_f_analog_2']
fig,axes = plt.subplots(1,2,figsize=(10,10),dpi=200)

for i, sn in enumerate(sig_names):
    df_meancorr = get_corr_spatial_distribution(xr_corr, df_metrics, sn)
    
    sns.heatmap(df_meancorr,cmap='vlag',ax=axes[i])
    axes[i].invert_yaxis()
    axes[i].set_title(sn)
    axes[i].set_ylabel('Depth um')

fig.tight_layout()
fig.savefig(soutput.corr_dist_plot)

# %%
