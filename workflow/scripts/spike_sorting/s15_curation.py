#%%
import os, sys
from pathlib import Path
from pprint import pprint 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bombcell as bc
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config

# %%
(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [config.debug_folder + r'/processed/df_bombcell.pkl'],
  'curation')

#%%
df_metrics = pd.read_pickle(sinput.df_quality_metrics)

#%%
ks_dir =Path(sinput.df_quality_metrics).parent/'kilosort4'/'ProbeA'
save_path = ks_dir/'bombcell'

#%% load the recording info
df_info = pd.read_csv(ks_dir.parent/'rec_prop_ProbeA.csv')

# extract the recording data path
root_dir = Path(os.environ['SESSION_ROOT_DIR']).parents[1]/'Data'
path = Path(df_info.iloc[0].full_path)
recording_path = Path(root_dir).joinpath(*path.parts[7:])


#%% load default parameters

raw_file = recording_path/'continuous.dat'
meta_file = recording_path.parents[1]/'structure.oebin'

param = bc.get_default_parameters(ks_dir, 
                                  raw_file=raw_file,
                                  meta_file=meta_file,
                                  kilosort_version=4)

param['maxRPVviolations'] =0.2 # align with kilosort default

# %%
(
    quality_metrics,
    param,
    unit_type,
    unit_type_string,
    figures,
) = bc.run_bombcell(
    ks_dir, save_path, param, return_figures=True,
)

# Conert quality metric to dataframe
quality_metrics_table = pd.DataFrame(quality_metrics)
quality_metrics_table.insert(0, 'Bombcell_unit_type', unit_type_string)

'''
Note: bombcell by default use the criteria by Hill et al 2011 for classifying MUA vs single-unit.
Hill et al does not use ISI violation for classification as used by Kilosort, but instead try to estimate the False discovery rate from ISI violation.
Its main assumption is that the contamination process is independent, which may not be true. Also the FDR depends on the firing rate.
See Vincent and Economo (2024) for a detailed discussion about the relation betwene ISI and FDR.
'''

#%%
# Compute ephys properties for cell type classification
ephys_param = bc.get_ephys_parameters(ks_dir)

# Compute all ephys properties - now defaults to ks_dir/bombcell
ephys_properties, ephys_param = bc.run_all_ephys_properties(ks_dir, ephys_param, save_path=save_path)

#%%
# Cell type classification with automatic plot generation
# Specify brain region: 'cortex' or 'striatum'
'''

'''
brain_region = 'striatum'  # Change this to 'cortex' for cortical data. Striatum and cortex are the only two options for now. 

# the classification is region specific
print(f"Classifying {brain_region} neurons...")
cell_types = bc.classify_and_plot_brain_region(ephys_properties, ephys_param, brain_region)

#%%
Show classification results
if cell_types is not None:
    # Create combined results table
    ephys_df = pd.DataFrame(ephys_properties)
    ephys_properties_table = ephys_df.copy()
    ephys_properties_table['cell_type'] = cell_types
   
    # print(ephys_properties_table)
    
#%% Merge ables
df_bombcell = quality_metrics_table.merge(ephys_properties_table, left_on='phy_clusterID', right_on='unit_id')


#%%

df_bombcell.attrs.update(param)
df_bombcell.attrs.update(ephys_param)

#%%
df_bombcell.to_pickle(soutput.df_bombcell)


# %% Create the figures
for n in figures.keys():
    if isinstance(figures[n], list):
        for i,fig in enumerate(figures[n]):
            fig.savefig(Path(soutput.curation_plots)/f'{n}_{i}.png', dpi=200)
    else:
        figures[n].savefig(Path(soutput.curation_plots)/f'{n}.png', dpi=200)

# #%% For investigation
# gui = bc.unit_quality_gui(
#     ks_dir=ks_dir,
#     quality_metrics=quality_metrics,
#     unit_types=unit_type,
#     param=param,
#     save_path=save_path,
# )


# #%%
# import numpy as np
# import bombcell.loading_utils as lu
# import bombcell.quality_metrics as qm

# # 1. Load the ephys data
# # This returns spike times in samples, cluster IDs, and template amplitudes
# spike_times_samples, spike_clusters, _, template_amplitudes, _, _, _ = lu.load_ephys_data(ks_dir)

# # 2. Convert spike times to seconds
# sample_rate = param['ephys_sample_rate']
# spike_times_seconds = spike_times_samples / sample_rate

# # 3. Select a specific unit to debug (e.g., unit 0)
# unit_id = 10  # <--- Change this to the unit ID you want to test
# unit_mask = spike_clusters == unit_id

# these_spike_times = spike_times_seconds[unit_mask]
# these_amplitudes = template_amplitudes[unit_mask]

# # 4. Define time chunks
# # If you aren't using time chunks in your main run, just use the start and end of the recording
# min_time = np.min(spike_times_seconds)
# max_time = np.max(spike_times_seconds)

# if param.get('computeTimeChunks', False):
#     delta_time = param.get('deltaTimeChunk', 360)
#     time_chunks = np.arange(min_time, max_time, delta_time)
#     if time_chunks[-1] < max_time:
#         time_chunks = np.append(time_chunks, max_time)
# else:
#     # Just one big chunk covering the whole recording
#     time_chunks = np.array([min_time, max_time])

# # 5. Call the function
# # Note: Ensure param has keys like 'tauR_valuesMin', 'tauR_valuesMax', 'tauC', etc.
# param['plotDetails'] = True
# fraction_RPVs, num_violations = qm.fraction_RP_violations(
#     these_spike_times, 
#     these_amplitudes, 
#     time_chunks, 
#     param,
#     return_per_bin=False
# )

# print(f"Fraction RPVs for unit {unit_id}:")
# print(fraction_RPVs)

# #%%
# plt.figure(figsize=(12, 2))
# plt.eventplot(these_spike_times, lineoffsets=0, linelengths=0.8, color='black')
# plt.xlim([0,10])
# plt.xlabel('Time (s)')
# plt.ylabel('Unit')
# plt.title(f'Raster plot for unit {unit_id}')
# plt.tight_layout()
# plt.show()