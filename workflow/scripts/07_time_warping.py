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
from workflow.scripts import settings
import trialexp.process.pyphotometry.linear_modelling as lm
from pathlib import Path
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [settings.debug_folder + '/processed/xr_photom_timewarped.nc'],
  'time_warping')

# %% Load data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
xr_photometry = xr.open_dataset(sinput.xr_photometry, engine = 'h5netcdf')


# %% Parameters
signal2analyze = ['analog_1_df_over_f', 'analog_2_df_over_f']

extraction_specs = {
    'hold_for_water' : {
        'event_window': [-500,500],
        'order': 'only',
        'padding':500
    },
    'bar_off': {
        'event_window':[-200,200],
        'order': 'last_before_first',
        'dependent_event': 'spout',
        'padding': 200
    },
    'spout': {
        'event_window':[-500,300],
        'order': 'first',
        'padding': 100
    },
    'lick': {
        'event_window':[-100,500],
        'order': 'first_after_last',
        'dependent_event': 'US_end_timer',
        'padding': 0
    }
}

#%% Time warping to align events

xr_conditions = xr.Dataset.from_dataframe(df_conditions)

xa_list = []
for signal_var in signal2analyze:
    xa = lm.time_warp_data(df_events_cond, 
                           xr_photometry[signal_var], 
                           extraction_specs, 
                           xr_photometry.attrs['sampling_rate'])
    
    xa_list.append(xa)
    
xr_warped = xr.merge([xr_conditions, *xa_list])

#%% Save data

xr_warped.to_netcdf(soutput.xr_timewarpped, engine='h5netcdf')


#%% Plot figures
outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
for var in signal2analyze:
    unique_outcome = np.unique(xr_warped.trial_outcome)
    fig, axes = plt.subplots(len(outcome2plot),1,figsize=(10,4*len(outcome2plot)))
    for outcome, ax in zip(outcome2plot, axes):
        xr2plot = xr_warped.sel(trial_nb = xr_warped.trial_outcome.isin(outcome))
        lm.plot_warpped_data(xr2plot, var, extraction_specs, ax=ax)
        
    fig.savefig(Path(soutput.figure_dir)/f'{var}_timewarp.png', dpi=200)


