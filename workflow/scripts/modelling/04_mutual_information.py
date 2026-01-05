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

#%% add in coefficient of variance information
# cv = ephys_utils.get_cell_mean_cv(xr_all_trials['spikes_FR_session'], axis=1)
# fr = ephys_utils.get_cell_mean_fr(xr_all_trials['spikes_FR_session'], axis=1)

# xr_all_trials['cv'] = xr.DataArray(cv, dims=['cluID'], coords={'cluID': xr_all_trials.cluID})
# xr_all_trials['fr'] = xr.DataArray(fr, dims=['cluID'], coords={'cluID': xr_all_trials.cluID})

# # Filter cells based on firing rate and coefficient of variation criteria
# mask = (xr_all_trials['fr'] > 1) #doesn't quite make sense to use cv as criteira here, as cell response differently for different trials
# xr_all_filtered = xr_all_trials.sel(cluID=mask)

# print(f'Original number of cells: {len(xr_all_trials.cluID)}')
# print(f'Filtered number of cells: {len(xr_all_filtered.cluID)}')

#%%
# # the combined dataset contains all possible trial_nb in the combined data, that's why some trial_nb are nan
# we need to filter it down to the valid data only, otherwise there will be too many nan

xr_session = xr_all_trials
xr_session = xr_session.sel(trial_nb = (xr_session.trial_outcome!='nan'))

save_files = [soutput.ach_model, soutput.da_model]
signal2analyze_list = ['zscored_df_over_f', 'zscored_df_over_f_analog_2','lick_rate']


#%% calculate MI on specific event window
for sig_var in signal2analyze_list:
    extraction_specs = eval(xr_session.attrs['extraction_specs'])
    event_win = mi.extract_event_windows(extraction_specs, xr_session)
    xr_mi = mi.calculate_mi_per_event(xr_session, event_win)

# %% split into different windows and calculate the MI



#%%
# sns.set_context('paper')
mi.plot_top_mi_cells(xr_mi, xr_session,'zscored_df_over_f_analog_2', extraction_specs, event='spout', n_cells=5)
# %%
