'''
Do time warping on spike sorted data

'''

'''
Perform time warping to align events across trials
'''

#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.plotting_utils import annotate_trial_number, plot_and_handler_error, plot_pyphoto_heatmap
from trialexp.process.pyphotometry.utils import *
import xarray as xr
import pandas as pd 
import seaborn as sns 
from matplotlib import pyplot as plt 
import numpy as np
import settings
import trialexp.process.pyphotometry.linear_modelling as lm
from pathlib import Path
import json
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + '/processed/xr_spikes_timewarped.nc'],
  'spike_timewarp')

# %% Load data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
xr_spike_fr = xr.open_dataset(sinput.xr_spikes_fr, engine = 'h5netcdf')
xr_corr = xr.open_dataset(sinput.xr_corr, engine = 'h5netcdf')
xr_photom_timewarpped = xr.open_dataset(sinput.xr_timewarpped, engine = 'h5netcdf')

# %% Parameters
signal2analyze = ['spikes_FR_session', 'spikes_zFR_session']
signal2analyze = [s for s in signal2analyze if s in xr_spike_fr.data_vars]

trigger, extraction_specs, outcome2plot = lm.get_warping_specs(df_events_cond, df_conditions, 'params/timewarp_spec.json')


#%% Time warping to align events

xr_conditions = xr.Dataset.from_dataframe(df_conditions)

xa_list = []
for signal_var in signal2analyze:
    xa, interp_results_list = lm.time_warp_data(df_events_cond, 
                           xr_spike_fr[signal_var], 
                           extraction_specs, 
                           trigger,
                           1000/xr_spike_fr.attrs['bin_duration'],
                           verbose=False)
                        
    
    xa_list.append(xa)
    
#add the interp result for later plotting
df_interp_res = pd.DataFrame(interp_results_list)
df_interp_res['trial_nb'] = xa.trial_nb
df_interp_res = df_interp_res.set_index('trial_nb')
xr_interp_res = df_interp_res.to_xarray()


#%% Save data
xr_warped = xr.merge([xr_conditions, xr_interp_res, *xa_list])
xr_warped.to_netcdf(soutput.xr_timewarpped, engine='h5netcdf')

#%% check for valid trials
valid_trials = np.all(~np.isnan(xr_warped['spikes_FR_session'].data[:,:,0]),axis=1)
print('Ratio of valid trials:', np.sum(valid_trials)/len(valid_trials))

#%% Plot presentative figures from each events


def plot_extrem_corr_timewarp(xr_corr, xr_spike_timewarp, xr_photom_timewarp, 
                              photom_signal_name,  spike_signal_name, event_name,trial_outcome='success', mode='abs'):
    '''
    Plot the average photometry signal together with the average firing rate for the largest correlations.

    Parameters:
    - xr_corr (xarray.DataArray): Xarray data array containing the correlation coefficients.
    - xr_spike_fr_interp (xarray.Dataset): Xarray dataset containing the interpolated spike firing rate data.
    - xr_session (xarray.Dataset): Xarray dataset containing the session data.
    - evt_name (str): Name of the event.
    - sig_name (str): Name of the signal.
    - mode (str, optional): How to sort the coefficient. Can be 'desc', 'asc', or 'abs'. Defaults to 'abs'.
        for 'abs', the absolute value will be sorted descendingly

    Returns:
    - None

    '''

    # find the largest correlation
    photom_data = np.squeeze(xr_photom_timewarp[photom_signal_name].data)
    fr = xr_spike_timewarp[spike_signal_name].data


    c = xr_corr[event_name+'_'+photom_signal_name].sel(trial_outcome=trial_outcome).data # event x 

    if mode == 'abs':
        idx = np.argmax(np.abs(c).data,axis=1) # the lag with the large correlation
        extrema_corr = c[np.arange(c.shape[0]),idx] #advance indexing do not work on xarray directly
        sorted_idx = np.argsort(np.abs(extrema_corr))[::-1]
    elif mode == 'asc':
        idx = np.argmin(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)
    else:
        idx = np.argmax(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)[::-1]
        
    lag = xr_corr.lag.data
    max_corr_loc = lag[idx]
    


    # plot average photometry signal together with time warped firing rate
    fig, axes = plt.subplots(3,1,figsize=(8,3*3))
    label1 = None
    label2 = None
    for i,ax in enumerate(axes.flat):
        ax2 = ax.twinx()
    
        if i == len(axes.flat)-1:
            label1='unit firing'
            label2='photmetry'
        
        x = np.nanmean(fr[:,:,sorted_idx[i]], axis=0)
        y = np.nanmean(photom_data,axis=0)

        lm.plot_warpped_data(xr2plot, var, extraction_specs, trigger, ax=ax)

        # shift = int(max_corr_loc[sorted_idx[i]]/(1000/50))
        # ax.plot(xr_spike_timewarp.time, x, label=label1)
        # ax.set_ylabel('Firing rate (Hz)')

        # ax2.plot(xr_photom_timewarp.time, y,'r', label=label2)
        # ax2.set_ylabel('dF/F')
        # ax.set_title(f'Largest corr = {extrema_corr[sorted_idx[i]]:.2f} at {max_corr_loc[sorted_idx[i]]:.1f}ms')
    
    fig.tight_layout()
    fig.legend()
    
    return fig


fig = plot_extrem_corr_timewarp(xr_corr, xr_warped, xr_photom_timewarpped, 'zscored_df_over_f', 
                           'spikes_zFR_session', 'bar_off', mode='abs')

# for var in signal2analyze:
#     unique_outcome = np.unique(xr_warped.trial_outcome)
#     fig, axes = plt.subplots(len(outcome2plot),1,figsize=(10,4*len(outcome2plot)))
    
#     if type(axes) is not np.ndarray:
#         axes =[axes]
        
#     for outcome, ax in zip(outcome2plot, axes):
#         xr2plot = xr_warped.sel(trial_nb = xr_warped.trial_outcome.isin(outcome))
#         lm.plot_warpped_data(xr2plot, var, extraction_specs, trigger, ax=ax)
        
#     fig.tight_layout()
#     fig.savefig(Path(soutput.figure_dir)/f'{var}_timewarp.png', bbox_inches='tight', dpi=200)

# %%
