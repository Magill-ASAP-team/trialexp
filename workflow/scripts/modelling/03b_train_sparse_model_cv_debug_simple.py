'''
Train sklearn baseline model for sparse encoding with 5-fold cross-validation
(Simplified version without time shifts for debugging)
'''

#%%
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import trialexp.process.ephys.utils as ephys_utils
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
import pickle
import trialexp.process.model.pytorch_decomposition as decomp
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
import xarray as xr
from scipy.signal import savgol_filter
from loguru import logger
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
n_folds = 10
random_state = 42


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

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(np.arange(n_valid_trials))):
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

    # ===== Train model using sklearn baseline =====
    logger.info("Training sklearn baseline model on train fold...")

    code_train, info_train, reconstruction_train, _ = \
        decomp.sparse_encode_sklearn_baseline(
            target=train_data['target_stack_smooth'],
            dictionary=train_data['dict_atoms_smooth'],
            sparsity_weight=1e-2,
            algorithm='lasso_lars',
            verbose=True
        )

    train_r2 = info_train['variance_explained']
    logger.info(f"Train R²: {train_r2:.4f}")

    # ===== Predict on test fold =====
    logger.info("Predicting on test fold with sklearn baseline...")

    # Use the same sklearn baseline on test data
    code_test, info_test, reconstruction_test, _ = \
        decomp.sparse_encode_sklearn_baseline(
            target=test_data['target_stack_smooth'],
            dictionary=test_data['dict_atoms_smooth'],
            code = code_train,
            sparsity_weight=1e-2,
            algorithm='lasso_lars',
            verbose=True
        )

    test_prediction = reconstruction_test
    test_r2 = info_test['variance_explained']

    logger.info(f"Test R²: {test_r2:.4f}")
    logger.info(f"Train-Test gap: {train_r2 - test_r2:.4f}")

    # Compute additional metrics
    test_mse = info_test['reconstruction_loss']
    logger.info(f"Test MSE: {test_mse:.6f}")
    logger.info(f"Train MSE: {info_train['reconstruction_loss']:.6f}")
    logger.info(f"Test non-zero coefficients: {info_test['n_nonzero']}")
    logger.info(f"Train non-zero coefficients: {info_train['n_nonzero']}")

    # ===== Store OOF predictions =====
    # Reshape predictions back to (n_test_trials × T)
    n_test_trials = len(test_idx)

    test_pred_trials = test_prediction.reshape(n_test_trials, T)
    test_true_trials = test_data['target_stack_smooth'].reshape(n_test_trials, T)

    # Store in OOF arrays (using original trial indices)
    for i, orig_idx in enumerate(test_idx):
        oof_predictions[orig_idx] = test_pred_trials[i]
        oof_ground_truth[orig_idx] = test_true_trials[i]

    # Store fold metadata
    fold_results.append({
        'fold_idx': fold_idx,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mse': info_train['reconstruction_loss'],
        'test_mse': test_mse,
        'n_nonzero_train': info_train['n_nonzero'],
        'n_nonzero_test': info_test['n_nonzero'],
        'code_train': code_train
    })

# ===== STAGE 3: Calculate Global OOF R² =====
logger.info(f"\n{'='*80}")
logger.info("Stage 3: Computing global out-of-fold metrics...")
logger.info(f"{'='*80}")

# Verify all trials were filled
assert not np.any(np.isnan(oof_predictions)), "Not all trials were filled in OOF predictions!"
assert not np.any(np.isnan(oof_ground_truth)), "Not all trials were filled in OOF ground truth!"

oof_pred_flat = oof_predictions.ravel()
oof_true_flat = oof_ground_truth.ravel()

oof_mse = np.mean((oof_pred_flat - oof_true_flat) ** 2)
oof_var = np.var(oof_true_flat)
oof_r2 = 1 - (oof_mse / oof_var)

# Calculate per-fold statistics
fold_train_r2_mean = np.mean([f['train_r2'] for f in fold_results])
fold_train_r2_std = np.std([f['train_r2'] for f in fold_results])
fold_test_r2_mean = np.mean([f['test_r2'] for f in fold_results])
fold_test_r2_std = np.std([f['test_r2'] for f in fold_results])

logger.info(f"\n{'='*60}")
logger.info("CROSS-VALIDATION RESULTS")
logger.info(f"{'='*60}")
logger.info(f"Global OOF R²: {oof_r2:.4f}")
logger.info(f"Mean fold train R²: {fold_train_r2_mean:.4f} ± {fold_train_r2_std:.4f}")
logger.info(f"Mean fold test R²: {fold_test_r2_mean:.4f} ± {fold_test_r2_std:.4f}")
logger.info(f"Train-test gap: {fold_train_r2_mean - fold_test_r2_mean:.4f}")
logger.info(f"{'='*60}\n")

# ===== STAGE 4: Save results =====
save_file = save_files[0]  # Using first save file for debugging
logger.info(f"Saving results to {save_file}")

with open(save_file, 'wb') as f:
    pickle.dump({
        'cv_type': 'kfold',
        'n_folds': n_folds,
        'random_state': random_state,
        'oof_predictions': oof_predictions,
        'oof_ground_truth': oof_ground_truth,
        'oof_r2': oof_r2,
        'oof_mse': oof_mse,
        'fold_results': fold_results,
        'fold_train_r2_mean': fold_train_r2_mean,
        'fold_train_r2_std': fold_train_r2_std,
        'fold_test_r2_mean': fold_test_r2_mean,
        'fold_test_r2_std': fold_test_r2_std,
        # Metadata
        'signal2analyze': signal2analyze,
        'trial_outcome': trial_outcome,
        'trial_nb': trial_nb,
        'n_valid_trials': n_valid_trials,
        'timepoints_per_trial': T,
        'n_neurons': atoms.shape[1],
        'sampling_rate': sampling_rate
    }, f)

logger.info(f"Completed processing {signal2analyze}\n")

#%% Visualization of last fold
fig, ax = plt.subplots(1,1, figsize=(12, 10), sharex=True)

# Plot last fold's test results
last_fold_idx = fold_results[-1]['fold_idx']
last_test_idx = fold_results[-1]['test_idx']

# Get predictions for last fold from OOF arrays
last_fold_pred = oof_predictions[last_test_idx]
last_fold_true = oof_ground_truth[last_test_idx]

ax.plot(last_fold_pred.ravel(),'k', label='Prediction')
ax.plot(last_fold_true.ravel(),'r', label='Ground Truth')
ax.legend()
ax.set_title(f'Fold {last_fold_idx+1} Test Set Predictions')
ax.set_xlabel('Timepoints')
ax.set_ylabel('Signal')
