'''
Do partial informationa decomposition to study influence of DA/ACh on neural firing rate
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
import itertools
from sklearn.feature_selection import mutual_info_regression

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/modelling.smk',
  [config.debug_folder + r'/processed/xr_mi.nc'],
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
signal2analyze_list = ['zscored_df_over_f', 'zscored_df_over_f_analog_2','lick_rate']

#%%
fr = xr_session['spikes_FR_session'].data
photom_da = xr_session['zscored_df_over_f_analog_2'].data  # trial x time
photom_ach = xr_session['zscored_df_over_f'].data 
fr = fr.transpose([2, 0, 1]) # cluID x trial x time

# filter invalid trials
mask_idx = (~np.isnan(fr[0, :, 0])) & (~np.isnan(photom_da[:,0]))
# Filter valid trials
fr = fr[:, mask_idx, :]
photom_da = photom_da[mask_idx,:]
photom_ach = photom_ach[mask_idx,:]

#%%
from idtxl.bivariate_pid import BivariatePID
from idtxl.data import Data
from idtxl.idtxl_utils import discretise  

alph = 5

x1 = discretise(photom_ach.ravel(), numBins=alph)
x1 = x1.reshape( photom_ach.shape[0],-1)

x2 = discretise(photom_da.ravel(), numBins=alph)
x2 = x2.reshape(photom_da.shape[0],-1)

y = discretise(fr.reshape(fr.shape[0],-1).T, numBins=alph) # expect replication x variable
y = y.T.reshape(fr.shape)
#%%
# plt.plot(fr[0,0,:])
# plt.plot(y[0,0,:])
#%%
# Note: calculation of PD is very computationally expensive and grow expontential according to the size of the alphabet
# too slow to do any permutation test
all_signals = np.array([x1.T, x2.T, y[0,:,:].T]).astype(int) # should be process x time x trial

data = Data(all_signals, dim_order='psr', normalise=False) # have to disable normalized

settings = {
    # 'nm_bins': alph,            # Number of bins for discretization (e.g., Low, Med, High)
    # 'discretisation': 'equiwidth', # Or 'equal_count' (quantiles)
    # 'n_perm_pid': 100,       # Number of permutations for statistical significance
    'pid_estimator': 'SydneyPID',
    'alph_s1': alph, #alphabet size
    'alph_s2': alph,
    'alph_t': alph,
    'alpha':0.05,
    'num_reps': 63,
    'max_iters': 1000,
    'max_unsuc_swaps_row_parm': 60,
}
pid_estimator = BivariatePID()
results = pid_estimator.analyse_single_target(data=data, target=2, sources=[0, 1], settings=settings)
#%%
# from idtxl.stats import unq_against_surrogates, syn_shd_against_surrogates  
# unq_results = unq_against_surrogates(pid_estimator, data)  
# syn_shd_results = syn_shd_against_surrogates(pid_estimator, data)


#%%
target_info = results.get_single_target(2)
unique_ach = target_info['unq_s1']
unique_da = target_info['unq_s1']
redundancy = target_info['shd_s1_s2']
synergy = target_info['syn_s1_s2']

print(f"Unique DA: {unique_da:.4f} bits")
print(f"Unique ACh: {unique_ach:.4f} bits")
print(f"Redundancy: {redundancy:.4f} bits")
print(f"Synergy: {synergy:.4f} bits")

#%% Use another library to try

import numpy as np
from entropy_estimators import continuous
from scipy.stats import multivariate_normal

def get_mvn_data(total_rvs, dimensionality=2, scale_sigma_offdiagonal_by=1., total_samples=1000):
    data_space_size = total_rvs * dimensionality

    # initialise distribution
    mu = np.random.randn(data_space_size)
    sigma = np.random.rand(data_space_size, data_space_size)
    # sigma = 1. + 0.5*np.random.randn(data_space_size, data_space_size)

    # ensures that sigma is positive semi-definite
    sigma = np.dot(sigma.transpose(), sigma)

    # scale off-diagonal entries -- might want to change that to block diagonal entries
    # diag = np.diag(sigma).copy()
    # sigma *= scale_sigma_offdiagonal_by
    # sigma[np.diag_indices(len(diag))] = diag

    # scale off-block diagonal entries
    d = dimensionality
    for ii, jj in itertools.product(list(range(total_rvs)), repeat=2):
        if ii != jj:
            sigma[d*ii:d*(ii+1), d*jj:d*(jj+1)] *= scale_sigma_offdiagonal_by

    # get samples
    samples = multivariate_normal(mu, sigma).rvs(total_samples)

    return [samples[:,ii*d:(ii+1)*d] for ii in range(total_rvs)]

k=5
norm='max'
X, Y, Z = get_mvn_data(total_rvs=3,
                        dimensionality=2,
                        scale_sigma_offdiagonal_by=1.,
                        total_samples=10000)

# solutions
# analytic = continuous.get_pmi_mvn(X, Y, Z)
# naive    = continuous.get_pmi(X, Y, Z, k=k, normalize=normalize, norm=norm, estimator='naive')
results       = continuous.get_pid(X[:,0], Y[:,0], Z[:,0], k=k, normalize=normalize, norm=norm)
#%%
(synergy, unique_ach, unique_da, redundancy) = continuous.get_pid(photom_ach.ravel(),
                                                                  photom_da.ravel(),
                                                                  fr[0].ravel(), 
                                                                  k=3)

#%%

def calculate_pid_per_event(xr_session, event_win, fr_var ='spikes_FR_session'):
    t = xr_session.time
    mi_list = []
    evt_list = []
    
    da_sig=xr_session['zscored_df_over_f_analog_2']
    ach_isg = xr_session['zscored_df_over_f']
    
    for (evt, win) in event_win.items():
        if (win[1] - win[0]):  # make sure there is data
            mask = (t >= win[0]) & (t <= win[1])
            xr_region = xr_session.sel(time=mask)
            
            fr = xr_region[fr_var].data  # trial x time x cluID
            
            photom_da = xr_region['zscored_df_over_f_analog_2'].data  # trial x time
            photom_ach = xr_region['zscored_df_over_f'].data 
            
            fr_stack, photom_stack_da, _ = mi.prepare_data_for_mi(fr, photom_da)
            fr_stack, photom_stack_ach, _ = mi.prepare_data_for_mi(fr, photom_ach)
            
            photo_joint = np.column_stack([photom_stack_da, photom_stack_ach])

            fr_cell = fr_stack[0]
            # print(photom_stack_da.shape)
            mi_joint = mutual_info_regression(photo_joint, fr_cell, n_jobs=10)
            mi_da = mutual_info_regression(photom_stack_da.reshape(-1,1), fr_cell, n_jobs=10)
            mi_ach = mutual_info_regression(photom_stack_da.reshape(-1,1), fr_cell, n_jobs=10)
            print(mi_joint)
            interaction_info = mi_joint - (mi_da + mi_ach)
            
            if interaction_info > 0:
                print(f"Synergy Dominates: {interaction_info}")
            else:
                print(f"Redundancy Dominates: {interaction_info}")
            
            mi_list.append(mi)
            evt_list.append(evt)
    
    mi_list = np.stack(mi_list)
    xr_mi = xr.DataArray(mi_list, dims=['event', 'cluID'], 
                         coords={'event': evt_list, 'cluID': xr_session.cluID})
    return xr_mi

xr_mi = xr.Dataset()
# calculate MI on specific event window
extraction_specs = eval(xr_session.attrs['extraction_specs'])
event_win = mi.extract_event_windows(extraction_specs, xr_session)
xa_mi = calculate_pid_per_event(xr_session,event_win)

    
#%%
    

#%%

#%% plot figures
for sig_var, evt in itertools.product(signal2analyze_list, xr_mi.event.data):
    logger.info(f'Processing {evt} of {sig_var}')
    fig = mi.plot_top_mi_cells(xa_mi, xr_session,sig_var, extraction_specs, event=evt, n_cells=5)
    
    fig.savefig(Path(soutput.figures_dir)/f'mi_{evt}_{sig_var}.png',dpi=200)
    
# %% save
xr_mi.to_netcdf(soutput.xr_mi)