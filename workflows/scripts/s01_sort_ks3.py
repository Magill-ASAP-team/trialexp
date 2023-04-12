'''
Script to create the session folder structure
'''
#%%
import os
import warnings

from pathlib import Path

import pandas as pd

from snakehelper.SnakeIOHelper import getSnake
from workflows.scripts import settings
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from spikeinterface.core import select_segment_recording
import spikeinterface.full as si
from trialexp.process.ephys.spikesort import sort


#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
  [settings.debug_folder + r'/processed/spike_sorting.done'],
  'spike_sorting')


# %%
# session_path = '/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/by_sessions/reaching_go_spout_bar_nov22/kms058-2023-03-25-184034'
# # rec_path = 
sorter_name = 'kilosort3'
# rec_properties_path = Path(session_path) / 'ephys' / 'rec_properties.csv'
rec_properties_path = Path(sinput.rec_properties)
rec_properties = pd.read_csv(rec_properties_path, index_col= None)

output_folder = rec_properties_path.parent / 'sorting'

# Only select longest syncable recordings to sort
idx_to_sort = rec_properties[rec_properties.longest == True].index.values

# root_data_path = r'/home/MRC.OX.AC.UK/phar0732/ettin/'
root_data_path = r'/home/MRC.OX.AC.UK/ndcn1330/ettin/Julien'

# %%
for idx_rec in idx_to_sort:
    exp_nb = rec_properties.exp_nb.iloc[idx_rec]
    rec_nb = rec_properties.rec_nb.iloc[idx_rec]
    AP_stream = rec_properties.AP_stream.iloc[idx_rec]
    
    # symplifying folder names for each probe
    if 'ProbeA' in AP_stream:    
        probe_folder_name = 'ProbeA'
    elif 'ProbeB' in AP_stream:
        probe_folder_name = 'ProbeB'
    else:
        raise ValueError(f'invalid probe name rec: {rec_properties_path.parent}')
    
    
    ephys_path = Path(rec_properties.full_path.iloc[idx_rec]).parent.parent.parent.parent.parent
    relative_ephys_path = os.path.join(*ephys_path.parts[5:])
    ephys_path = os.path.join(root_data_path, relative_ephys_path)
    
    recordings = se.read_openephys(ephys_path, block_index=exp_nb, stream_name=AP_stream)
    recording = select_segment_recording(recordings, segment_indices= int(rec_nb-1))
    print('Now sorting' ,recording)
    sorter_specific_params = {
        # 'n_jobs': 24, 
        # 'total_memory': 512000000000, 
        # 'chunk_size': None, 
        # 'chunk_memory': 12800000000,
        'chunk_duration': '10s', 
        'progress_bar': True}

    sorting = ss.run_sorter(
            sorter_name = sorter_name,
            recording = recording, 
            output_folder = output_folder / 'output'/ probe_folder_name,
            remove_existing_folder = True, 
            delete_output_folder = False, 
            verbose = True,
            **sorter_specific_params)

    sorting.save(folder= output_folder / probe_folder_name)

# %% loading back the data for testing

sorting_re = si.load_extractor(output_folder /probe_folder_name)
# %%
