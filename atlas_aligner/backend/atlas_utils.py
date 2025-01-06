#%%
import fastapi
from enum import Enum
from typing import Optional
from trialexp.process.folder_org.utils import build_session_info_cohort
from trialexp.process.anatomy.utils import load_ccf_data, shift_trajectory_depth, get_region_boundaries
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
coords = probe_ccf[0]['trajectory_coords'].astype(float)

#%% atlas
atlas, structure_tree = load_ccf_data(Path('/mnt/Magill_Lab/Julien/ASAP/software/allenccf'))

# %%
# load the channel location and limit the trajectory length
channel_position =np.load(df_sel.path/'processed/kilosort4/ProbeA/channel_positions.npy')
max_depth = channel_position.max(axis=0)[1]
# %%
shifted_coords1 = shift_trajectory_depth(coords, 0, length=max_depth)
trajectory_areas = get_region_boundaries(shifted_coords1, atlas, structure_tree)
trajectory_areas[['acronym','depth_start','depth_end']]
# %%

