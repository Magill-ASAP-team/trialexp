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


app = fastapi.FastAPI()
app.state.df_session_info = df_session_info

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/sessions/')
async def get_sessions(cohort: Optional[str] = None, animal_id: Optional[str] = None):
    query_str = ''
    
    if cohort:
        query_str += f"cohort=='{cohort}'"

    if animal_id:
        query_str += f'and animal_id=="{animal_id}"'
    
    df = app.state.df_session_info
    
    if len(query_str) > 0:
        df = df.query(query_str)
        
    # only return session with sorted neuropixel data
    df = df.query('neuropixels_sorted==True')
        
    return {"session_id": df.session_id.unique().tolist()}

@app.get('/cohorts')
async def get_cohort():
    df = app.state.df_session_info
    return {"cohorts": df.cohort.unique().tolist()}


@app.get('/animal_id')
async def get_animal_id(cohort: Optional[str] = None):
    df = app.state.df_session_info
    if cohort:
        df = df.query(f"cohort=='{cohort}'")
    return {"animal_id": df.animal_id.unique().tolist()}


@app.get('/trajectory')
async def get_trajectory(session_id: str):
    df = app.state.df_session_info
    df = df.query(f"session_id=='{session_id}'")
    df = df.query('neuropixels_sorted==True')
    df = df.iloc[0]
    local_results = Path(root_path)/df.cohort/'histology'/df.animal_id/'RGB'/'Processed'
    probe_ccf = sio.loadmat(str(local_results/'probe_ccf.mat'), simplify_cells=True)['probe_ccf']
    coords = probe_ccf[0]['trajectory_coords'].astype(float)
    atlas, structure_tree = load_ccf_data(Path('/mnt/Magill_Lab/Julien/ASAP/software/allenccf'))
    # shifted_coords = shift_trajectory_depth(coords, -1000)
    trajectory_areas = get_region_boundaries(coords, atlas, structure_tree)
    return trajectory_areas[['acronym','depth_start','depth_end']].to_dict(orient='records')
# #%%

# root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
# df_session_info = build_session_info_cohort(root_path)
# # %%
# df_sesssion_info = df_session_info.query('neuropixels_sorted==True')
# #%%
# df_sel = df_session_info.query('animal_id=="TT011"').iloc[0]
# # %%
# # match with the track localization results
# local_results = Path(root_path)/df_sel.cohort/'histology'/df_sel.animal_id/'RGB'/'Processed'
# # %%

# probe_ccf = sio.loadmat(str(local_results/'probe_ccf.mat'), simplify_cells=True)['probe_ccf']
# # %%
# probe_ccf[0]['trajectory_areas']
# # %%
# # trajectory coords is in ap, dv, ml
# # coords is [start, end] in axis coordinate
# # ref see https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
# coords = probe_ccf[0]['trajectory_coords'].astype(float)

# #%% atlas
# atlas, structure_tree = load_ccf_data(Path('/mnt/Magill_Lab/Julien/ASAP/software/allenccf'))

# # Example usage
# shifted_coords = shift_trajectory_depth(coords, -1000)
# trajectory_areas = get_region_boundaries(shifted_coords, atlas, structure_tree)
# trajectory_areas[['acronym','depth_start','depth_end']]
# # %%
