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
import json
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [settings.debug_folder + '/processed/xr_photom_timewarped.nc'],
  'time_warping')

# %% Load data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
xr_photometry = xr.open_dataset(sinput.xr_photometry, engine = 'h5netcdf')


# %% Parameters
signal2analyze = ['zscored_df_over_f', 'zscored_df_over_f_analog_2','zscored_df_over_f_analog_3']

with open('params/timewarp_spec.json') as f:
    specs = json.load(f)

task_name = df_events_cond.attrs['task_name']
trigger = df_events_cond.attrs['triggers'][0]

if task_name in ['pavlovian_spontanous_reaching_oct23',
                 'pavlovian_reaching_Oct26',
                 'pavlovian_spontanous_reaching_march23',
                 'pavlovian_spontanous_reaching_oct23']:
    
    extraction_specs = specs['spontanous_reaching']
    outcome2plot = df_conditions.trial_outcome.unique()
    
elif task_name in ['reaching_go_spout_bar_VR_Dec23']:
    extraction_specs = specs['reaching_go_spout_bar_reward']
    outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
    
elif task_name in ['reaching_go_spout_bar_mar23',
                   'reaching_go_spout_bar_june05',
                   'reaching_go_spout_bar_nov22',
                   'reaching_go_spout_bar_apr23']:
    extraction_specs = specs['reaching_go_spout_bar']
    outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
    
elif task_name in ['reaching_go_spout_incr_break2_nov22']:
    extraction_specs = specs['break2']
    outcome2plot = df_conditions.trial_outcome.unique()
else:
    extraction_specs = specs['default']
    #update the trigger
    extraction_specs[trigger] = extraction_specs.pop('trigger')
    outcome2plot = df_conditions.trial_outcome.unique()


#%% Time warping to align events

xr_conditions = xr.Dataset.from_dataframe(df_conditions)

xa_list = []
for signal_var in signal2analyze:
    xa, interp_results_list = lm.time_warp_data(df_events_cond, 
                           xr_photometry[signal_var], 
                           extraction_specs, 
                           trigger,
                           xr_photometry.attrs['sampling_rate'],
                           verbose=False)
    
    xa_list.append(xa)
    
#add the interp result for later plotting
df_interp_res = pd.DataFrame(interp_results_list)
df_interp_res['trial_nb'] = xa.trial_nb
df_interp_res = df_interp_res.set_index('trial_nb')
xr_interp_res = df_interp_res.to_xarray()

xr_warped = xr.merge([xr_conditions, xr_interp_res, *xa_list])

#%% Save data

xr_warped.to_netcdf(soutput.xr_timewarpped, engine='h5netcdf')


#%% Plot the time wrapped data
for var in signal2analyze:
    unique_outcome = np.unique(xr_warped.trial_outcome)
    fig, axes = plt.subplots(len(outcome2plot),1,figsize=(10,4*len(outcome2plot)))
    
    if type(axes) is not np.ndarray:
        axes =[axes]
        
    for outcome, ax in zip(outcome2plot, axes):
        xr2plot = xr_warped.sel(trial_nb = xr_warped.trial_outcome.isin(outcome))
        lm.plot_warpped_data(xr2plot, var, extraction_specs, trigger, ax=ax)
        
        
    fig.savefig(Path(soutput.figure_dir)/f'{var}_timewarp.png', bbox_inches='tight', dpi=200)
