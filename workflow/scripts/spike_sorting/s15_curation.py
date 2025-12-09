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
  [config.debug_folder + r'/processed/df_celltypes.pkl'],
  'cell_classification')

#%%
ks_dir =Path(sinput.df_quality_metrics).parent/'kilosort4'/'ProbeA'
save_path = ks_dir/'bombcell'

#%% load the recording info
df_info = pd.read_csv(ks_dir.parent/'rec_prop_ProbeA.csv')
# %%
root_dir = Path(os.environ['SESSION_ROOT_DIR']).parents[1]/'Data'
print(root_dir)
path = Path(df_info.iloc[0].full_path)
recording_path = Path(root_dir).joinpath(*path.parts[7:])
print(recording_path)
#%%

raw_file = recording_path/'continuous.dat'
meta_file = recording_path.parents[1]/'structure.oebin'

param = bc.get_default_parameters(ks_dir, 
                                  raw_file=raw_file,
                                  meta_file=meta_file,
                                  kilosort_version=4)

print("Bombcell parameters:")
pprint(param)
# %%
(
    quality_metrics,
    param,
    unit_type,
    unit_type_string,
) = bc.run_bombcell(
    ks_dir, save_path, param
)
# %%
# quality metric values
quality_metrics_table = pd.DataFrame(quality_metrics)
quality_metrics_table.insert(0, 'Bombcell_unit_type', unit_type_string)
quality_metrics_table    
#%%
boolean_quality_metrics_table = bc.make_qm_table(
    quality_metrics, param, unit_type_string
)
boolean_quality_metrics_table
#%%
gui = bc.unit_quality_gui(
    ks_dir=ks_dir,
    quality_metrics=quality_metrics,
    unit_types=unit_type,
    param=param,
    save_path=save_path,
)
#%%
# Compute ephys properties for cell type classification
ephys_param = bc.get_ephys_parameters(ks_dir)

# Compute all ephys properties - now defaults to ks_dir/bombcell
ephys_properties, ephys_param = bc.run_all_ephys_properties(ks_dir, ephys_param, save_path=save_path)
#%%
# Cell type classification with automatic plot generation
# Specify brain region: 'cortex' or 'striatum'
brain_region = 'striatum'  # Change this to 'cortex' for cortical data. Striatum and cortex are the only two options for now. 

print(f"Classifying {brain_region} neurons...")
cell_types = bc.classify_and_plot_brain_region(ephys_properties, ephys_param, brain_region)

#%%
# Cell type classification with automatic plot generation
# Specify brain region: 'cortex' or 'striatum'
brain_region = 'striatum'  # Change this to 'cortex' for cortical data. Striatum and cortex are the only two options for now. 

print(f"Classifying {brain_region} neurons...")
cell_types = bc.classify_and_plot_brain_region(ephys_properties, ephys_param, brain_region)