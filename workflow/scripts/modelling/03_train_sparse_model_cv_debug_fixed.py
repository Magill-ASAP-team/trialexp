'''
Train pytorch model for sparse encoding - FIXED VERSION (single fold for debugging)
'''

#%%
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import trialexp.process.ephys.utils as ephys_utils
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
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

# Train-Val-Test split configuration (8:1:1)
random_state = 42
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
val_early_stop_patience = 3  # Early stopping patience

#%%
signal2analyze = signal2analyze_list[0]

logger.info(f"\n{'='*80}")
logger.info(f"Processing {signal2analyze} with train-val-test split ({int(train_ratio*100)}:{int(val_ratio*100)}:{int(test_ratio*100)})")
logger.info(f"{'='*80}\n")

# ===== STAGE 1: Prepare full dataset (filters NaN trials internally) =====
logger.info("Stage 1: Preparing full dataset (NEW trial-stacked format)...")
full_data = decomp.prepare_data_for_encoding_trials(xr_session, signal2analyze)

# Extract arrays (already filtered, includes baseline atom)
atoms = full_data['atoms_norm_smooth']  # (time × n_neurons+1 × n_valid_trials) - includes baseline!
target = full_data['target_norm_smooth']  # (time × n_valid_trials)
lick_rate = full_data['lick_rate']
trial_outcome = full_data['trial_outcome']
trial_nb = full_data['trial_nb']

n_valid_trials = atoms.shape[2]
T = atoms.shape[0]  # timepoints per trial

logger.info(f"Valid trials after filtering: {n_valid_trials}")
logger.info(f"Timepoints per trial: {T}")
logger.info(f"Number of neurons (including baseline): {atoms.shape[1]}")

# Transpose to model input format: (n_neurons+1, time, n_trials)
dict_atoms = atoms.transpose([1, 0, 2])

# ===== STAGE 2: Split data into train/val/test (8:1:1) =====
logger.info(f"\nStage 2: Splitting data into train/val/test ({int(train_ratio*100)}:{int(val_ratio*100)}:{int(test_ratio*100)})...")

# Create indices for all trials
all_indices = np.arange(n_valid_trials)

# First split: separate test set (10%)
train_val_idx, test_idx = train_test_split(
    all_indices,
    test_size=test_ratio,
    random_state=random_state,
    shuffle=True
)

# Second split: separate validation from training (1/9 of remaining = 10% of total)
# val_ratio / (train_ratio + val_ratio) = 0.1 / 0.9 = 0.111...
train_inner_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=val_ratio / (train_ratio + val_ratio),
    random_state=random_state,
    shuffle=True
)

logger.info(f"\n{'-'*60}")
logger.info(f"Data split: {len(train_inner_idx)} train, {len(val_idx)} val, {len(test_idx)} test")
logger.info(f"Proportions: {len(train_inner_idx)/n_valid_trials:.1%} train, {len(val_idx)/n_valid_trials:.1%} val, {len(test_idx)/n_valid_trials:.1%} test")
logger.info(f"{'-'*60}")

# ===== Prepare train, validation, and test data (SIMPLE INDEXING - no concatenation!) =====
dict_train = dict_atoms[:, :, train_inner_idx]  # (n_neurons+1, time, n_train)
target_train = target[:, train_inner_idx]        # (time, n_train)

dict_val = dict_atoms[:, :, val_idx]    # (n_neurons+1, time, n_val)
target_val = target[:, val_idx]          # (time, n_val)

dict_test = dict_atoms[:, :, test_idx]    # (n_neurons+1, time, n_test)
target_test = target[:, test_idx]          # (time, n_test)

logger.info(f"\nData shapes (NEW format):")
logger.info(f"  Train dict: {dict_train.shape}")
logger.info(f"  Train target: {target_train.shape}")
logger.info(f"  Val dict: {dict_val.shape}")
logger.info(f"  Val target: {target_val.shape}")
logger.info(f"  Test dict: {dict_test.shape}")
logger.info(f"  Test target: {target_test.shape}")

# ===== Train model (NEW wrapper function) =====
logger.info("\nTraining model on train fold (using sparse_encode_pytorch_with_shift_trials)...")
init_shifts = np.random.normal(100, 1, dict_train.shape[0])

code_train, info_train, reconstruction_train, model_train = \
    decomp.sparse_encode_pytorch_with_shift_trials(
        target=target_train,
        dictionary=dict_train,
        validation_target=target_val,
        validation_dictionary=dict_val,
        val_early_stop_patience=val_early_stop_patience,
        val_early_stop_metric='r2',
        save_best_model=True,
        verbose_validation=True,
        max_shift_ms=[-200, 200],
        init_shift_ms=init_shifts,
        sampling_rate=sampling_rate,
        n_iterations=10,
        sparsity_weight=5e-3,
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
logger.info(f"\nTrain R²: {train_r2:.4f}")
logger.info(f"Train code shape: {model_train.code.shape}")

# Log validation results
if info_train['val_r2_history']:
    best_val_r2 = info_train['best_val_metric']
    final_val_r2 = info_train['val_r2_history'][-1]
    logger.info(f"Best Val R²: {best_val_r2:.4f} at iteration {info_train['best_val_iteration']+1}")
    logger.info(f"Final Val R²: {final_val_r2:.4f}")
    if info_train['val_early_stopped']:
        logger.info(f"Training stopped early after {info_train['n_iterations_completed']} iterations")

# ===== Predict on test fold (FROZEN MODEL - no training!) =====
logger.info("\nPredicting on test fold (frozen model)...")
logger.info(f"Test data has {len(test_idx)} trials")

# Use the new evaluation function
test_results = decomp.evaluate_sparse_model_on_test_data(
    model_train=model_train,
    dict_test=dict_test,
    target_test=target_test,
    sampling_rate=sampling_rate,
    max_shift_ms=[-100, 100],
    device='cuda'
)

# Extract results
reconstruction_test = test_results['reconstruction']
test_r2 = test_results['test_r2']
test_mse = test_results['test_mse']
test_var = test_results['test_var']
model_test = test_results['model_test']

logger.info(f"\nTest code shape: {model_test.code.shape}")
logger.info(f"Test reconstruction shape: {reconstruction_test.shape}")
logger.info(f"Learned shifts (ms): min={info_train['shifts_ms'].min():.1f}, max={info_train['shifts_ms'].max():.1f}, mean={info_train['shifts_ms'].mean():.1f}")

logger.info(f"\n{'='*60}")
logger.info("RESULTS")
logger.info(f"{'='*60}")
logger.info(f"Train R²: {train_r2:.4f}")
if info_train['val_r2_history']:
    logger.info(f"Val R²:   {info_train['val_r2_history'][-1]:.4f} (best: {info_train['best_val_metric']:.4f})")
logger.info(f"Test R²:  {test_r2:.4f}")
logger.info(f"Train-Test gap: {train_r2 - test_r2:.4f}")
if info_train['val_r2_history']:
    logger.info(f"Train-Val gap: {train_r2 - info_train['val_r2_history'][-1]:.4f}")
logger.info(f"\nTest MSE: {test_mse:.6f}")
logger.info(f"Train MSE: {info_train['reconstruction_loss']:.6f}")
logger.info(f"{'='*60}")

# Diagnostic checks
logger.info(f"\n{'='*60}")
logger.info("DIAGNOSTIC CHECKS")
logger.info(f"{'='*60}")
logger.info(f"Prediction range: [{reconstruction_test.min():.3f}, {reconstruction_test.max():.3f}]")
logger.info(f"Target range: [{target_test.min():.3f}, {target_test.max():.3f}]")
logger.info(f"Prediction mean: {reconstruction_test.mean():.3f} ± {reconstruction_test.std():.3f}")
logger.info(f"Target mean: {target_test.mean():.3f} ± {target_test.std():.3f}")

if np.isnan(test_r2):
    logger.error("❌ Test R² is NaN!")
elif test_r2 < -1:
    logger.warning(f"⚠ Test R² is very negative ({test_r2:.4f}), model is performing worse than predicting the mean")
elif test_r2 < 0:
    logger.warning(f"⚠ Test R² is negative ({test_r2:.4f}), model is performing worse than predicting the mean")
else:
    logger.info(f"✓ Test R² is {test_r2:.4f}")

# Clear GPU memory
del model_train, model_test
torch.cuda.empty_cache()

#%% Visualization
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Number of timepoints to plot (use all data or limit to 2000)
n_plot_test = min(2000, reconstruction_test.size)
n_plot_train = min(2000, reconstruction_train.size)

# ===== Top panel: Test Set =====
ax = axes[0]
# IMPORTANT: ravel in column-major (Fortran) order to match (trials, time) concatenation order
# reconstruction_test is (time, trials), we need to flatten as [trial0_t0, trial0_t1, ..., trial1_t0, ...]
ax.plot(reconstruction_test.T.ravel()[:n_plot_test], 'r', label='Prediction', alpha=0.7, linewidth=1)
ax.plot(target_test.T.ravel()[:n_plot_test], 'k', label='Ground Truth', alpha=0.5, linewidth=1)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_title(f'Test Set Predictions (R² = {test_r2:.4f}, n={len(test_idx)} trials)', fontsize=12, fontweight='bold')
ax.set_xlabel('Timepoints', fontsize=10)
ax.set_ylabel('Normalized Signal', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, min(1000, n_plot_test)])

# Add text box with statistics
textstr = f'MSE: {test_mse:.4f}\nMean: {reconstruction_test.mean():.3f} ± {reconstruction_test.std():.3f}'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ===== Bottom panel: Train Set =====
ax = axes[1]
ax.plot(reconstruction_train.T.ravel()[:n_plot_train], 'r', label='Prediction', alpha=0.7, linewidth=1)
ax.plot(target_train.T.ravel()[:n_plot_train], 'k', label='Ground Truth', alpha=0.5, linewidth=1)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_title(f'Train Set Predictions (R² = {train_r2:.4f}, n={len(train_inner_idx)} trials)', fontsize=12, fontweight='bold')
ax.set_xlabel('Timepoints', fontsize=10)
ax.set_ylabel('Normalized Signal', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, min(1000, n_plot_train)])

# Add text box with statistics
train_mse = info_train['reconstruction_loss']
textstr = f'MSE: {train_mse:.4f}\nMean: {reconstruction_train.mean():.3f} ± {reconstruction_train.std():.3f}'
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure
output_path = '/home/MRC.OX.AC.UK/ndcn1330/code/trialexp/debug_prediction_plot.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
logger.info(f"\nVisualization saved to: {output_path}")
plt.close()

# ===== Additional visualization: Validation learning curves =====
if info_train['val_r2_history']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: R² curves
    ax = axes[0]
    iterations = np.arange(1, len(info_train['train_r2_history']) + 1)
    ax.plot(iterations, info_train['train_r2_history'], 'o-', label='Train R²', linewidth=2, markersize=6)
    ax.plot(iterations, info_train['val_r2_history'], 's-', label='Validation R²', linewidth=2, markersize=6)

    # Mark best validation iteration
    best_iter = info_train['best_val_iteration'] + 1
    best_val = info_train['best_val_metric']
    ax.axvline(best_iter, color='gray', linestyle='--', alpha=0.5, label=f'Best Val (iter {best_iter})')
    ax.plot(best_iter, best_val, 'r*', markersize=15, label=f'Best Val R²: {best_val:.4f}')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('R² Score', fontsize=11)
    ax.set_title('Learning Curves: R² vs Iteration', fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(max(info_train['train_r2_history']), max(info_train['val_r2_history'])) * 1.1])

    # Right panel: MSE curves
    ax = axes[1]
    ax.plot(iterations, info_train['val_mse_history'], 's-', label='Validation MSE', linewidth=2, markersize=6, color='C1')
    ax.axvline(best_iter, color='gray', linestyle='--', alpha=0.5, label=f'Best Val (iter {best_iter})')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Validation MSE vs Iteration', fontsize=12, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save learning curves
    output_path_curves = '/home/MRC.OX.AC.UK/ndcn1330/code/trialexp/debug_learning_curves.png'
    plt.savefig(output_path_curves, dpi=150, bbox_inches='tight')
    logger.info(f"Learning curves saved to: {output_path_curves}")
    plt.close()
# %%
