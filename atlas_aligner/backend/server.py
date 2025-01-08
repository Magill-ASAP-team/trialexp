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
from datetime import datetime
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
async def get_trajectory(session_id: str, shift:int = 0):
    df = app.state.df_session_info
    df = df.query(f"session_id=='{session_id}'")
    df = df.query('neuropixels_sorted==True')
    df = df.iloc[0]
    local_results = Path(root_path)/df.cohort/'histology'/df.animal_id/'RGB'/'Processed'
    probe_ccf = sio.loadmat(str(local_results/'probe_ccf.mat'), simplify_cells=True)['probe_ccf']
    
    
    with open(local_results/'probe_names.txt') as f:
        probe_names = f.readlines()
        probes = []
        for p in probe_names:
            try: 
                probes.append(datetime.strptime(p.strip(), '%d/%m/%Y').date())
            except:
                probes.append(p)

    probes_name = np.array(probes)


    # # match the experiment datetime
    probe_idx = np.where(probes_name == df.expt_datetime.date())[0]
    if len(probe_idx) ==0:
        return None
    atlas, structure_tree = load_ccf_data(Path('/mnt/Magill_Lab/Julien/ASAP/software/allenccf'))
    
    channel_position =np.load(df.path/'processed/kilosort4/ProbeA/channel_positions.npy')
    
    probe_coords = trajectory2probe_coords(probe_ccf[probe_idx[0]], channel_position)
    
    shifted_coords = shift_trajectory_depth(probe_coords, shift)
    trajectory_areas = get_region_boundaries(shifted_coords, atlas, structure_tree)
    trajectory_areas = trajectory_areas.sort_values('depth_start')
    trajectory_areas['track_date'] = str(probes_name[probe_idx[0]])
    return trajectory_areas[['acronym','depth_start','track_date','depth_end','name']].to_dict(orient='records')


@app.get('/cell_metrics')
async def get_firing_rate(session_id:str):
    df = app.state.df_session_info
    df = df.query(f"session_id=='{session_id}'")
    df = df.query('neuropixels_sorted==True')
    
    if len(df)>0:
        df = df.iloc[0]
        df_quality_metrics = pd.read_pickle(df.path/'processed/df_quality_metrics.pkl')
        df_fr = df_quality_metrics.groupby('ks_chan_pos_y')['firing_rate'].agg(['mean','count']).reset_index()
        return df_fr.to_dict(orient='list')
    
