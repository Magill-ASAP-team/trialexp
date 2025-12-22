'''
Train pytorch model for sparse encoding with time shift
shuffling analysis
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
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/modelling.smk',
  [config.debug_folder + r'/processed/da_sparse_encode.pkl'],
  'train_sparse_model')

#%% load data
xr_spikes_all = xr.open_dataset(sinput.xr_timewarpped)
xr_photom_warp = xr.load_dataset(sinput.xr_photom_timewarped)

#%% Merge Neuropixels with photometry
sampling_rate = 1000/np.diff(xr_photom_warp['time'])[0]
print(f'Photometry sample rate: {sampling_rate} Hz')

xr_spikes_all = xr_spikes_all.interp(time = xr_photom_warp.time) # make sure the time axis is the same

xr_all_trials = xr_spikes_all.merge(xr_photom_warp)

#%% add in coefficient of variance information
cv = ephys_utils.get_cell_mean_cv(xr_all_trials['spikes_FR_session'], axis=1)
fr = ephys_utils.get_cell_mean_fr(xr_all_trials['spikes_FR_session'], axis=1)

xr_all_trials['cv'] = xr.DataArray(cv, dims=['cluID'], coords={'cluID': xr_all_trials.cluID})
xr_all_trials['fr'] = xr.DataArray(fr, dims=['cluID'], coords={'cluID': xr_all_trials.cluID})

# Filter cells based on firing rate and coefficient of variation criteria
mask = (xr_all_trials['fr'] > 1) #doesn't quite make sense to use cv as criteira here, as cell response differently for different trials
xr_all_filtered = xr_all_trials.sel(cluID=mask)

print(f'Original number of cells: {len(xr_all_trials.cluID)}')
print(f'Filtered number of cells: {len(xr_all_filtered.cluID)}')

#%%
# # the combined dataset contains all possible trial_nb in the combined data, that's why some trial_nb are nan
# we need to filter it down to the valid data only, otherwise there will be too many nan

xr_session = xr_all_filtered
xr_session = xr_session.sel(trial_nb = (xr_session.trial_outcome!='nan'))

save_files = [soutput.ach_model, soutput.da_model]
signal2analyze_list = ['zscored_df_over_f', 'zscored_df_over_f_analog_2']

# for target_sensor, signal2analyze in zip(target_sensor, signal2analyze_list):
# Extract atoms and target for all trials
# extract the appropriate signal for each trial outcome 

n_shuffle = 1

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

for save_file, signal2analyze in zip(save_files, signal2analyze_list):
    
    atoms = xr_session['spikes_FR_session'].data # original dimen: trial x time x cluID
    atoms = atoms.transpose([1,2,0]) # time x cluID x trial
    
    target = xr_session[signal2analyze].data
    
    shuffle_results = []
    
    for n in range(n_shuffle):
        print(f"Shuffle iteration {n+1}/{n_shuffle}")
        
        target = target.T #time x trial
        lick_rate = xr_session['lick_rate'].data
        lick_rate = lick_rate.T

        # Use the function
        mask_idx = (~np.isnan(atoms[0, 0, :])) & (~np.isnan(target[0, :]))
        prepared_data = decomp.prepare_data_for_encoding(xr_session, signal2analyze, mask_idx)

        dict_atoms_smooth = prepared_data['dict_atoms_smooth']
        target_stack_smooth = prepared_data['target_stack_smooth']
        target_stack = prepared_data['target_stack']
        atoms = prepared_data['atoms']
        target = prepared_data['target']
        event_time = prepared_data['event_time']
        cluID_atom = prepared_data['cluID_atom']
        lick_rate = prepared_data['lick_rate']

        # Shuffle the trial dimension in target
        n_trials = target.shape[1]
        shuffled_trial_indices = np.random.permutation(n_trials)
        target = target[:, shuffled_trial_indices]
        target_stack_smooth = target_stack_smooth[:, shuffled_trial_indices]
        target_stack = target_stack[:, shuffled_trial_indices]
        lick_rate = lick_rate[:, shuffled_trial_indices]

        # check the average signal to make sure the smoothing is appropriate
        target_stack_smooth_trial = target_stack_smooth.reshape(target.shape[1],-1)

        trial_mask = (xr_session.trial_outcome[mask_idx] == 'success')



        init_shifts = np.random.normal(100, 1, dict_atoms_smooth.shape[0])

        code_final, info, reconstruction, model = decomp.sparse_encode_pytorch_with_shift(
            target=target_stack_smooth,
            dictionary=dict_atoms_smooth,  # unshifted patterns
            max_shift_ms=200.0,  # Fixed: changed from list to float
            init_shift_ms= init_shifts,
            sampling_rate=sampling_rate,
            n_iterations=10,
            sparsity_weight=1e-3,
            n_steps_code=5000,
            n_steps_shift= 1000,
            sparsity_type='elastic_net',
            max_lr_shift=0.1,
            device='cuda',
            shift_print_step= 400,
            code_print_step=1000,
            early_stop_patience = 100,
            use_mlflow=True  # Track experiments
        )

        shuffle_results.append({
            'shuffle_idx': n,
            'shuffled_trial_indices': shuffled_trial_indices,
            'code_final': code_final,
            'target_stack_smooth': target_stack_smooth,
            'dict_atoms_smooth': dict_atoms_smooth,
            'reconstruction': reconstruction,
            'trial_outcome':  xr_session.trial_outcome[mask_idx],
            'event_time': event_time,
            'atoms': atoms,
            'target': target,
            'info': info,
            'model': model,
            'lick_rate': lick_rate,
            'cluID': cluID_atom,
            'trial_nb': xr_session.trial_nb[mask_idx],
        })
    
    with open(save_file,'wb') as f:
        pickle.dump({
            'n_shuffle': n_shuffle,
            'shuffle_results': shuffle_results
        }, f)
        