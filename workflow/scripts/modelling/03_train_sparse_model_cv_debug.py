'''
Train pytorch model for sparse encoding with 5-fold cross-validation
'''

#%%
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import trialexp.process.ephys.utils as ephys_utils
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
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
  [config.debug_folder + r'/processed/ach_sparse_encode_cv.pkl'],
  'train_sparse_model_cv')

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
# signal2analyze_list = ['zscored_df_over_f', 'zscored_df_over_f_analog_2']
signal2analyze_list = ['zscored_df_over_f_analog_2']

# CV configuration
n_folds = 2
random_state = 42

#%%
# full_data = decomp.prepare_data_for_encoding(xr_session, signal2analyze_list[1])

# #%%

# print(full_data['atoms_smooth'].shape) #103 cell
# plt.plot(full_data['atoms_smooth'][:,0,0])

# #%%
# print(full_data['atoms_stack'].shape) 
# plt.plot(full_data['atoms_stack'][0,:][:400])

# #%%
# print(full_data['atoms_norm_smooth'].shape) #103 cell
# plt.plot(full_data['atoms_norm_smooth'][:,2,0])

# #%%
# plt.plot(full_data['target_norm_smooth'][:,0])

#%%
train_data = decomp.prepare_fold_from_atoms_target(
    atoms, target, train_idx, lick_rate
)
test_data = decomp.prepare_fold_from_atoms_target(
    atoms, target, test_idx, lick_rate
)
#%%

# plt.plot(test_data['dict_atoms_smooth'][4,:][:232])
plt.plot(train_data['dict_atoms_smooth'][4,:][:232])

#%%
plt.plot(test_data['target_stack_smooth'][:,:232].T)
plt.plot(train_data['target_stack_smooth'][:,:232].T)

#%%
plt.plot(test_prediction.T[:232],'r')
plt.plot(test_data['target_stack_smooth'].T[:232])


test_mse = np.mean((test_prediction - test_data['target_stack_smooth']) ** 2)
test_var = np.var(test_data['target_stack_smooth'])
test_r2 = 1 - (test_mse / test_var)
print(test_r2)


#%%
signal2analyze = signal2analyze_list[0]

logger.info(f"\n{'='*80}")
logger.info(f"Processing {signal2analyze} with {n_folds}-fold cross-validation")
logger.info(f"{'='*80}\n")

# ===== STAGE 1: Prepare full dataset (filters NaN trials internally) =====
logger.info("Stage 1: Preparing full dataset...")
full_data = decomp.prepare_data_for_encoding(xr_session, signal2analyze)

# Extract arrays (already filtered)
atoms = full_data['atoms_norm_smooth']  # (time × cluID × n_valid_trials)
target = full_data['target_norm_smooth']  # (time × n_valid_trials)
lick_rate = full_data['lick_rate']
trial_outcome = full_data['trial_outcome']
trial_nb = full_data['trial_nb']

n_valid_trials = atoms.shape[2]
T = atoms.shape[0]  # timepoints per trial

logger.info(f"Valid trials after filtering: {n_valid_trials}")
logger.info(f"Timepoints per trial: {T}")
logger.info(f"Number of neurons: {atoms.shape[1]}")

# Initialize CV splitter
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

# Storage for OOF predictions
oof_predictions = np.full((n_valid_trials, T), np.nan)
oof_ground_truth = np.full((n_valid_trials, T), np.nan)
fold_results = []

# ===== STAGE 2: CV loop =====
logger.info(f"\nStage 2: Running {n_folds}-fold cross-validation...")

# for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.arange(n_valid_trials))):

(train_idx, test_idx)  = list(kfold.split(np.arange(n_valid_trials)))[0]

logger.info(f"\n{'-'*60}")
logger.info(f"Fold {fold_idx+1}/{n_folds}")
logger.info(f"Train trials: {len(train_idx)}, Test trials: {len(test_idx)}")
logger.info(f"{'-'*60}")

# ===== Prepare train and test data from trial subsets =====
train_data = decomp.prepare_fold_from_atoms_target(
    atoms, target, train_idx, lick_rate
)
test_data = decomp.prepare_fold_from_atoms_target(
    atoms, target, test_idx, lick_rate
)

# ===== Train model =====
logger.info("Training model on train fold...")
init_shifts = np.random.normal(100, 1, train_data['dict_atoms_smooth'].shape[0])

code_train, info_train, reconstruction_train, model_train = \
    decomp.sparse_encode_pytorch_with_shift(
        target=train_data['target_stack_smooth'],
        dictionary=train_data['dict_atoms_smooth'],
        max_shift_ms=[-200, 200],
        init_shift_ms=init_shifts,
        sampling_rate=sampling_rate,
        n_iterations=10,
        sparsity_weight=1e-3,
        n_steps_code=5000,
        n_steps_shift=1000,
        sparsity_type='elastic_net',
        max_lr_shift=0.1,
        device='cuda',
        shift_print_step=400,
        code_print_step=1000,
        early_stop_patience=100,
        use_mlflow=False
    )

train_r2 = info_train['variance_explained']
logger.info(f"Train R²: {train_r2:.4f}")

# ===== Predict on test fold =====
logger.info("Predicting on test fold...")
with torch.no_grad():
    model_train.eval()

    # Apply learned model to test dictionary
    test_dict_tensor = torch.FloatTensor(test_data['dict_atoms_smooth']).to('cuda')

    # Create new FourierShiftDictionary for test data
    test_shift_dict = decomp.FourierShiftDictionary(
        dictionary=test_dict_tensor,
        max_shift_ms=[-200, 200],
        sampling_rate=sampling_rate
    ).to('cuda')

    # Copy learned shifts
    test_shift_dict.shift_logits.data = model_train.shift_dictionary.shift_logits.data

    # Create test model
    model_test = decomp.SparseCodingWithShifts(
        shift_dictionary=test_shift_dict,
        n_neurons=test_dict_tensor.shape[0],
        activation_type='global'  # Match training
    ).to('cuda')

    # Copy learned code and activation
    model_test.code.data = model_train.code.data
    model_test.act.load_state_dict(model_train.act.state_dict())

    # Forward pass
    test_prediction = model_test().cpu().numpy()

    # Calculate test R²
    
    test_mse = np.mean((test_prediction - test_data['target_stack_smooth']) ** 2)
    test_var = np.var(test_data['target_stack_smooth'])
    test_r2 = 1 - (test_mse / test_var)

logger.info(f"Test R²: {test_r2:.4f}")
logger.info(f"Train-Test gap: {train_r2 - test_r2:.4f}")

# Clear GPU memory
# del model_train, model_test, test_dict_tensor, test_shift_dict
# torch.cuda.empty_cache()

