#%%
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from kilosort.io import load_ops
from spikeinterface.core.template import Templates  

#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/df_celltypes.pkl'],
  'cell_classification')

#%%
df_quality_metrics = pd.read_pickle(sinput.df_quality_metrics)
# %%
df_sel = df_quality_metrics[df_quality_metrics['peak_to_valley']<0.4]
#%%
np.array(df_quality_metrics.iloc[0].templates).shape

# %%

kilsort_folder = Path(sinput.df_quality_metrics).parent/'kilosort4/ProbeA'
ops = load_ops(kilsort_folder/'ops.npy')
nt = ops['nt']
fs = ops['fs']
templates_array = np.stack(df_quality_metrics.templates.values)

#%%

  
# Your template data (must be 3D: units x samples x channels)  
# templates_array = np.random.randn(3, 50, 8)  # 3 units, 50 samples, 8 channels  
nbefore = fs*(1/1000)  # 20 samples before peak  
  
templates = Templates(  
    templates_array=templates_array,  
    sampling_frequency=fs,  
    nbefore=nbefore  
)
#%%
from spikeinterface.core.template_tools import get_template_extremum_channel  
extremum_channels = get_template_extremum_channel(templates, peak_sign="both", outputs='index')  
print(extremum_channels)

# check with the maxwaveform
extre_idx = np.array(list(extremum_channels.values()))
maxcha = df_quality_metrics['maxWaveformCh'].values
#%%
print(extre_idx[:3])
print(maxcha[:3])
print(np.mean(np.abs(extre_idx-maxcha)))
print(np.mean(extre_idx==maxcha))
'''
85% of cluster show difference < 4 channel
'''
'''
[  2 158   0] extre
[2 1 0] max
'''
#%%
#plot spikes for comparison
# (533, 90, 384)

# %%
templates_kilo = np.load(kilsort_folder/'templates.npy')
ks_label = pd.read_csv(kilsort_folder/'cluster_KSLabel.tsv',sep='\t')
whitening_mat_inv = np.load(kilsort_folder/'whitening_mat_inv.npy')
amplitudes = np.load(kilsort_folder/'amplitudes.npy')


original_templates = np.zeros_like(templates_kilo)  
for t in range(templates_kilo.shape[1]):  # For each time point  
    original_templates[:, t, :] = templates_kilo[:, t, :] @ whitening_mat_inv.T

original_templates = original_templates[ks_label['KSLabel']=='good',:,:]
cluster_id = np.flatnonzero(ks_label['KSLabel']=='good')
clus = np.load(kilsort_folder/'spike_clusters.npy')

'''
by default, kilosort use -(nt//2), nt//2+1 to extract the spikes
spikeinterface uses 1ms before and 2ms after to extract the template

spikeinterface compute using random waveforms
maxchan is using the template from kilosort
template from kilsort is in the whitening space, so the spatial spread may be different
even in get_templates function the templates are scaled but still in whitening space
You need to recover the original by 
w = whitening_mat_inv @ w

Also spikeinterface return data in uV
But kilosort directly load the data in int16 format, so it may be missing a scaling factor
the channel gain for openephys is 0.19499999, which seem to make sense
'''

#%% Compare the template from spikeinterface and kilosort
cluster2plot = 10
fig, ax = plt.subplots()
scale = 0.19499999
ax.plot(templates_array[cluster2plot,:,extre_idx[cluster2plot]], label='spikeinterface')
ax.plot(original_templates[cluster2plot,:,extre_idx[cluster2plot]]*scale, label='kilosort')
fig.legend()

amplitudes[clus==cluster_id[cluster2plot]].mean()

#%%
cluster2plot = 5
fig, ax = plt.subplots()
ax.plot(templates_array[cluster2plot,:,extre_idx[cluster2plot]], label='extrem')
ax.plot(templates_array[cluster2plot,:,maxcha[cluster2plot]], label='maxchan')
fig.legend()