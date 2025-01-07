#%%
import fastapi
from enum import Enum
from typing import Optional
from trialexp.process.folder_org.utils import build_session_info_cohort
from trialexp.process.anatomy.utils import load_ccf_data, shift_trajectory_depth, get_region_boundaries, trajectory2probe_coords
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

#%%

root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
df_session_info = build_session_info_cohort(root_path)
# %%
df_sesssion_info = df_session_info.query('neuropixels_sorted==True')
#%%
df_sel = df_session_info.query('session_id=="TT013-2024-10-24-142256"').iloc[0]
# %%
# match with the track localization results
local_results = Path(root_path)/df_sel.cohort/'histology'/df_sel.animal_id/'RGB'/'Processed'
# %%

probe_ccf = sio.loadmat(str(local_results/'probe_ccf.mat'), simplify_cells=True)['probe_ccf']
# %%
probe_ccf[0]['trajectory_areas']
# %%
# trajectory coords is in ap, dv, ml
# coords is [start, end] in axis coordinate
# ref see https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858


channel_position =np.load(df_sel.path/'processed/kilosort4/ProbeA/channel_positions.npy')


probe_coords = trajectory2probe_coords(probe_ccf[0], channel_position)
#%% atlas
atlas, structure_tree = load_ccf_data(Path('/mnt/Magill_Lab/Julien/ASAP/software/allenccf'))


# %%
shifted_coords1 = shift_trajectory_depth(probe_coords, -619)
trajectory_areas = get_region_boundaries(shifted_coords1, atlas, structure_tree)
trajectory_areas = trajectory_areas.sort_values('depth_start')
trajectory_areas[['acronym','depth_start','depth_end']]
# %% Plot the fiing rate map

df_quality_metrics = pd.read_pickle(df_sel.path/'processed/df_quality_metrics.pkl')

#%%
df_fr = df_quality_metrics.groupby('ks_chan_pos_y')[['firing_rate']].mean().reset_index()

df_fr.to_dict(orient='list')
# %%
#TODO: we need to identify the rought location of the tip using the last point from the alignment
# find the lowest point on the DV axis and 

# %%

# %%
