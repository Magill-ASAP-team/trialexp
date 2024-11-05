'''
Do time warping on spike sorted data

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
import itertools
import trialexp.process.ephys.photom_correlation as pc
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + '/processed/xr_spikes_timewarped.nc'],
  'spike_timewarp')

# %% Load data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
xr_spike_fr = xr.open_dataset(sinput.xr_spikes_fr, engine = 'h5netcdf')
xr_corr = xr.load_dataset(sinput.xr_corr, engine = 'h5netcdf')
xr_photom_timewarpped = xr.load_dataset(sinput.xr_timewarpped, engine = 'h5netcdf')
xr_session = xr.load_dataset(sinput.xr_session, engine = 'h5netcdf')

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

# Get the names of the signal and event to plot
photom_vars = ['zscored_df_over_f', 'zscored_df_over_f_analog_2']
event2plot = [v.replace('_'+photom_vars[0],'') for v in xr_session.data_vars.keys() if v.endswith(photom_vars[0]) and not (v in photom_vars)]

#plot
for photom_signal_name, evt_name in itertools.product(photom_vars, event2plot):

    fig = pc.plot_extrem_corr_timewarp(extraction_specs, trigger, 
                                       xr_corr, xr_warped, xr_photom_timewarpped, photom_signal_name, 
                            'spikes_FR_session', 
                            evt_name, mode='abs')

    fig.savefig(Path(soutput.figure_dir)/f'timewarp_{evt_name}_{photom_signal_name}.png',dpi=200)
# %%
xr_spike_fr.close()