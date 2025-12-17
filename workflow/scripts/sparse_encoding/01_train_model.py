'''
Train pytorch model for sparse encoding with time shift
'''

#%%
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
import glob
from sklearn import preprocessing
import matplotlib.gridspec as gridspec
import trialexp.process.ephys.utils as ephys_utils
from sklearn.preprocessing import normalize
import torch
import pickle
import trialexp.process.model.pytorch_decomposition as decom

#%%
xr_spikes_all = xr.open_dataset(df2load.path/'processed/xr_spikes_timewarped.nc')

#%%
# load the time warp data
xr_photom_warp = xr.load_dataset(df2load.path/'processed'/'xr_photom_timewarped.nc')
sampling_rate = 1000/np.diff(xr_photom_warp['time'])[0]
print(f'Photometry sample rate: {sampling_rate} Hz')

xr_spikes_all = xr_spikes_all.interp(time = xr_photom_warp.time)

# Use all trials without splitting by outcome
xr_all_trials = xr_spikes_all.merge(xr_photom_warp)

#%% add in coefficient of variance information
cv = ephys_utils.get_cell_mean_cv(xr_all_trials['spikes_FR_session'], axis=1)
fr = ephys_utils.get_cell_mean_fr(xr_all_trials['spikes_FR_session'], axis=1)

xr_all_trials['cv'] = xr.DataArray(cv, dims=['cluID'], coords={'cluID': xr_all_trials.cluID})
xr_all_trials['fr'] = xr.DataArray(fr, dims=['cluID'], coords={'cluID': xr_all_trials.cluID}
                                   
#%%
# Filter cells based on firing rate and coefficient of variation criteria
mask = (xr_all_trials['fr'] > 1) #doesn't quite make sense to use cv as criteira here, as cell response differently for different trials
xr_all_filtered = xr_all_trials.sel(cluID=mask)

print(f'Original number of cells: {len(xr_all_trials.cluID)}')
print(f'Filtered number of cells: {len(xr_all_filtered.cluID)}')

#%%
# session_ids = xr2analyze.session_id
# xr_session = xr2analyze.sel(session_id = session_ids[0])
# # the combined dataset contains all possible trial_nb in the combined data, that's why some trial_nb are nan
# we need to filter it down to the valid data only, otherwise there will be too many nan

xr_session = xr_all_filtered
xr_session = xr_session.sel(trial_nb = (xr_session.trial_outcome!='nan'))
xr_session

# Extract atoms and target for all trials
# extract the appropriate signal for each trial outcome 

atoms = xr_session['spikes_FR_session'].data # original dimen: trial x time x cluID
atoms = atoms.transpose([1,2,0]) # time x cluID x trial

target = xr_session[signal2analyze].data
target = target.T #time x trial
lick_rate = xr_session['lick_rate'].data
lick_rate = lick_rate.T

# sometimes the photometry may stop first and result in nan trial
mask_idx = (~np.isnan(atoms[0,0,:])) & (~np.isnan(target[0,:]))

atoms = atoms[:,:,mask_idx]
target =target[:,mask_idx]
lick_rate = lick_rate[:, mask_idx]
event_time = xr_session.time
cluID_atom = xr_session.cluID.data

#%%
import pytorch_decomposition as decomp

# Original atoms
atoms_stack = atoms.transpose([1,2,0])
atoms_stack = atoms_stack.reshape(atoms_stack.shape[0], -1)


def normal_twoend(x):
    x = 2 * (x - x.min()) / (x.max() - x.min()) - 1 #normalize to [-1, 1]
    return x

# add in the intercept
baseline_atom = -np.ones((1, atoms_stack.shape[1])) 
dict_atoms = np.vstack([atoms_stack, baseline_atom])
dict_atoms = normalize(dict_atoms, axis=1)

target_stack = target.T.reshape(1, -1)
target_stack = normal_twoend(target_stack)

# smoothing
target_stack_smooth = savgol_filter( target_stack.ravel(), 21, 2).reshape(1,-1)
target_stack_smooth = normal_twoend(target_stack_smooth)
dict_atoms_smooth = savgol_filter( dict_atoms, 21, 2).reshape(dict_atoms.shape[0], -1)


# check the average signal to make sure the smoothing is appropriate
target_stack_smooth_trial = target_stack_smooth.reshape(target.shape[1],-1)

trial_mask = (xr_session.trial_outcome[mask_idx] == 'success')
plt.plot(event_time,target_stack_smooth_trial[trial_mask,:].mean(axis=0))

plt.figure()
plt.plot(np.arange(len(target_stack.ravel()))/50, target_stack.ravel(), alpha=0.5)
plt.plot(np.arange(len(target_stack_smooth.ravel()))/50, target_stack_smooth.ravel(),'r-')
plt.xlim([0, 50])

#%%

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

init_shifts = np.random.normal(100, 1, dict_atoms.shape[0])

code_final, info, reconstruction, model = decomp.sparse_encode_pytorch_with_shift(
    target=target_stack_smooth,
    dictionary=dict_atoms_smooth,  # unshifted patterns
    max_shift_ms=[-200, 200],
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

#%%
t = np.arange(len(reconstruction.ravel()))/50
fig,ax = plt.subplots(1,1,figsize=(6,3))
ax.plot(t, target_stack_smooth.T)
ax.plot(t, reconstruction.T,'r')
ax.set_xlim([100,150])

with open(f'pytorch_model_{target_sensor}_{session_id}_timewarp.pickle','wb') as f:
    pickle.dump({
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
    }, f)