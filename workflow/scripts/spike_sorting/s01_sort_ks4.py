'''
Script to compute cell metrics by CellExplorer from Kilosort3 results
'''
#%%
import os
import warnings

import shutil

from pathlib import Path
import numpy as np

import pandas as pd

from snakehelper.SnakeIOHelper import getSnake
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.core import select_segment_recording
from kilosort import run_kilosort
import settings
import torch
from trialexp.process.ephys.artifact_removal import filter_artifact_sensor
from kilosort.io import BinaryFiltered

#%% Load inputs
spike_sorting_done_path = str(Path(settings.debug_folder) / 'processed' / 'spike_sorting.done')
(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
 [spike_sorting_done_path], 'spike_sorting')

# %%

sorter_name = 'kilosort4'
verbose = True
rec_properties_path = Path(sinput.rec_properties)
session_path = rec_properties_path.parents[1]/'processed'
session_id = rec_properties_path.parents[1].stem
rec_properties = pd.read_csv(rec_properties_path, index_col = 0)
rec_properties['sorting_error'] = False
# Only select longest syncable recordings to sort
idx_to_sort = rec_properties[(rec_properties.syncable == True) & (rec_properties.longest==True)].index.values

root_data_path = os.environ['SORTING_ROOT_DATA_PATH']
temp_sorter_folder = Path(os.environ['TEMP_DATA_PATH']) /session_id
output_si_sorted_folder = Path(soutput.si_output_folder)

# %%
for idx_rec in idx_to_sort:

    AP_stream = rec_properties.AP_stream.iloc[idx_rec]
    recording_path = rec_properties.full_path[idx_rec]
    
    # symplifying folder names for each probe
    if 'ProbeA' in AP_stream:    
        probe_name = 'ProbeA'
    elif 'ProbeB' in AP_stream:
        probe_name = 'ProbeB'
    else:
        raise ValueError(f'invalid probe name rec: {rec_properties_path.parent}')

    # Define outputs folder, specific for each probe and sorter
    temp_output_sorter_specific_folder = temp_sorter_folder / sorter_name / probe_name

    # ephys_path = Path(rec_properties.full_path.iloc[idx_rec]).parents[4]
    
    # Maybe not the best method to get it
    # has introduced some bugs for forgotten reason related to folder changes
    # # TODO improve to join just before relative_ephys_path and root_data_path overlap
    # relative_ephys_path = os.path.join(*ephys_path.parts[5:])
    # ephys_path = os.path.join(root_data_path, relative_ephys_path)
    ephys_path = Path(rec_properties.full_path.iloc[idx_rec]).parts[-10:]
    
    recording_path = os.path.join(root_data_path, *ephys_path)
    
    if not (output_si_sorted_folder/probe_name).exists():
        (output_si_sorted_folder/probe_name).mkdir()
    
    # use kilosort4 directly
    device = torch.device('cuda:0')
    settings = {'data_dir': recording_path, 
                'n_chan_bin': 384, 
                'batch_size' : 30000*8, # 8*Fs for speeding up, for bad session, use a smaller batch_size
                'save_extra_vars': True,
                # 'tmax':1600,
                'results_dir': output_si_sorted_folder/probe_name}
    
    #####
    # artifact removal. Only activate this on bad session
    # BinaryFiltered.filter = filter_artifact_sensor 
    
    run_kilosort(settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat', device=device)
    
        
    rec2save = rec_properties.iloc[[idx_rec]].copy()
    # cannot save in the probe folder otherwise spikeinterface will complain
    rec2save.to_csv(output_si_sorted_folder/f'rec_prop_{probe_name}.csv', index=False) #also save the recording property



# %% free GPU memory
torch.cuda.empty_cache()