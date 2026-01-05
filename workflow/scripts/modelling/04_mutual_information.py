'''
Train pytorch model for sparse encoding with time shift
'''

#%%
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import trialexp.process.ephys.utils as ephys_utils
from sklearn.preprocessing import normalize
import torch
import pickle
import trialexp.process.model.pytorch_decomposition as decomp
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
import xarray as xr
from scipy.signal import savgol_filter
from loguru import logger
from pathlib import Path
import numpy as np
from trialexp.process.pyphotometry.linear_modelling import compute_ticks
import trialexp.process.model.mutual_info as mi 
import seaborn as sns
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/modelling.smk',
  [config.debug_folder + r'/processed/da_mutual_info.pkl'],
  'compute_mutual_information')

#%% load data
xr_spikes_all = xr.open_dataset(sinput.xr_timewarpped)
xr_photom_warp = xr.load_dataset(sinput.xr_photom_timewarped)

#%% Merge Neuropixels with photometry
sampling_rate = 1000/np.diff(xr_photom_warp['time'])[0]
print(f'Photometry sample rate: {sampling_rate} Hz')

xr_spikes_all = xr_spikes_all.interp(time = xr_photom_warp.time) # make sure the time axis is the same

xr_all_trials = xr_spikes_all.merge(xr_photom_warp)


xr_session = xr_all_trials
xr_session = xr_session.sel(trial_nb = (xr_session.trial_outcome!='nan'))

save_files = [soutput.ach_model, soutput.da_model]
signal2analyze_list = ['zscored_df_over_f', 'zscored_df_over_f_analog_2','lick_rate']

#%%
for sig_var in signal2analyze_list:
    # calculate MI on specific event window
    extraction_specs = eval(xr_session.attrs['extraction_specs'])
    event_win = mi.extract_event_windows(extraction_specs, xr_session)
    xr_mi = mi.calculate_mi_per_event(xr_session,event_win, photom_var=sig_var)
    fig = mi.plot_top_mi_cells(xr_mi, xr_session,'zscored_df_over_f_analog_2', extraction_specs, event='spout', n_cells=5)
# %%
