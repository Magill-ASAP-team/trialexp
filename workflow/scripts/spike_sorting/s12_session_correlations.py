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

from trialexp.process.ephys.utils import crosscorr_lag_range, plot_correlated_neurons
from trialexp.process.group_analysis.plot_utils import style_plot
from trialexp.process.pyphotometry.utils import *
import settings
from tqdm.auto import tqdm
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/df_cross_corr.pkl'],
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

var2compare = 'first_spout'

fr = xr_spike_fr_interp[f'spikes_FR.{var2compare}']
photom = xr_session[f'{var2compare}{photom_vars[0]}']

fr_data = fr.data
photom_data = np.squeeze(photom.data)

#%%
from scipy.stats import pearsonr

def calculate_pearson_lags(x, y, max_lag, lag_step=1):
    """
    Calculate Pearson correlation coefficients and lags between two signals.
    
    Improve speed by creating a lag matrix.
    Each row of the lag matrix corresponds to the original signal shifted by some lag.
    The shifted signals from all trials are flattened into one row.

    Args:
        x (ndarray): The first signal.
        y (ndarray): The second signal.
        max_lag (int): The maximum lag to consider.

    Returns:
        tuple: A tuple containing:
            - lags (ndarray): The array of lag values.
            - correlations (ndarray): The correlation matrix containing the auto and cross correlations.
            - corr (ndarray): The cross-correlation values.
     
    """
    lags = np.arange(-max_lag, max_lag + 1, lag_step)
    correlations = np.zeros(len(lags))
    # y = np.tile(y, (x.shape[0],1)) # expand into a matrix
    xm = np.zeros((len(lags),x.shape[0]*x.shape[1]))
    ym = np.zeros_like(xm)
    
    for i, lag in enumerate(lags):
        if lag < 0:
            shifted_x = x[:,:lag]
            shifted_y = y[:,-lag:]
        elif lag > 0:
            shifted_x = x[:,lag:]
            shifted_y = y[:,:-lag]
        else:
            shifted_x = x
            shifted_y = y
        
        # remove NAN data
        valid_idx = ~np.isnan(shifted_y.mean(axis=1))
        shifted_x = shifted_x[valid_idx,:].ravel()
        shifted_y = shifted_y[valid_idx,:].ravel() 
        
        assert len(shifted_x) == len(shifted_y), f'Length mismatch {len(shifted_x)} vs {len(shifted_y)}'
        xm[i, :len(shifted_x)] = shifted_x
        ym[i, :len(shifted_y)] = shifted_y
    
    correlations= np.corrcoef(xm, ym) #contains the auto and then cross correlation between variables
    halfw = len(correlations)//2
    corr = np.diag(correlations[halfw:, :halfw]) # the bottom quandrant is the cross-correlation
    return lags,correlations,corr

max_lags = 50
lag_step = 5
nlags = max_lags//lag_step
c = np.zeros((fr_data.shape[2], nlags*2+1, fr_data.shape[2]))
corr = np.zeros((fr_data.shape[2],nlags*2+1))
for i in tqdm(range(fr_data.shape[2])):
    lags,_, corr[i,:] = calculate_pearson_lags(fr_data[:,:,i], photom_data,max_lags, lag_step)
#%%    
plt.imshow(corr,aspect='auto');plt.colorbar()

#%%
evt_time = xr_spike_fr_interp.spk_event_time
evt_time_step = np.mean(np.diff(evt_time))

max_id = np.argsort(corr.min(axis=1))
plt.plot(lags*evt_time_step, corr[max_id[0],:])

fig, axes = plt.subplots(3,3,figsize=(3*3,3*3))


for i,ax in enumerate(axes.flat):
    ax2 = ax.twinx()

    ax.plot(evt_time, fr_data[:,:,max_id[i]].mean(axis=0), label='unit firing')
    ax2.plot(evt_time, np.nanmean(photom_data,axis=0),'r', label='photometry')

fig.tight_layout()
#%%
df_cross_corr = pd.DataFrame({
    'cluID': UIDs.values,
    'cross_corr': cross_corr.tolist()
})
df_cross_corr.to_pickle(soutput.df_cross_corr)
#%% Plot the first few cell with maximum cross correlation with photometry

fig = plot_correlated_neurons(cross_corr, xr_spike_session, lags, n_fig=8)
fig.savefig(soutput.corr_plot)

# %%
#TODO: event in specific type of trial
# snakemake --snakefile workflow/spikesort.smk -n ~/ettin/Julien/Data/head-fixed/by_sessions/reaching_go_spout_bar_nov22/kms058-2023-03-25-184034/processed/spike_workflow.done