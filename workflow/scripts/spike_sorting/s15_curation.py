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

# print("Bombcell parameters:")
# pprint(param)

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

#%%
# Compute ephys properties for cell type classification
ephys_param = bc.get_ephys_parameters(ks_dir)

# Compute all ephys properties - now defaults to ks_dir/bombcell
ephys_properties, ephys_param = bc.run_all_ephys_properties(ks_dir, ephys_param, save_path=save_path)

#%%
# Cell type classification with automatic plot generation
# Specify brain region: 'cortex' or 'striatum'
brain_region = 'striatum'  # Change this to 'cortex' for cortical data. Striatum and cortex are the only two options for now. 

# the classification is region specific
print(f"Classifying {brain_region} neurons...")
cell_types = bc.classify_and_plot_brain_region(ephys_properties, ephys_param, brain_region)

#%%
# Show classification results
if cell_types is not None:
    # Create combined results table
    ephys_df = pd.DataFrame(ephys_properties)
    ephys_properties_table = ephys_df.copy()
    ephys_properties_table['cell_type'] = cell_types
   
    # print(ephys_properties_table)
    
#%% Merge ables
df_bombcell = quality_metrics_table.merge(ephys_properties_table, left_on='phy_clusterID', right_on='unit_id')


#%%
df_bombcell.to_pickle(soutput.df_bombcell)

