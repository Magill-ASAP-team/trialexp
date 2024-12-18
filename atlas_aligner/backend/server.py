#%%
import fastapi
from enum import Enum
from typing import Optional
from trialexp.process.folder_org.utils import build_session_info_cohort
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import pandas as pd

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


#%%

root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
df_session_info = build_session_info_cohort(root_path)
# %%
df_sesssion_info = df_session_info.query('neuropixels_sorted==True')
#%%
df_sel = df_session_info.query('animal_id=="TT011"').iloc[0]
# %%
# match with the track localization results
local_results = Path(root_path)/df_sel.cohort/'histology'/df_sel.animal_id/'RGB'/'Processed'
# %%
import scipy.io as sio
probe_ccf = sio.loadmat(str(local_results/'probe_ccf.mat'), simplify_cells=True)['probe_ccf']
# %%
probe_ccf[0]['trajectory_areas']
# %%
# trajectory coords is in ap, dv, ml
# coords is [start, end] in axis coordinate
# ref see https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
coords = probe_ccf[0]['trajectory_coords'].astype(float)

#%% atlas

def load_ccf_data(base_directory):
    
    # The annoatation volume is a 3D volume with an index at each voxel
    # it present the row of in the structure_tree_safe_2017 table
    # the template volume contains the reference image of the brain (voxel with each element the brightness of the pixel)
    # the atlas is 1-based, its index corresponds to the row of the structure_tree_safe_2017 table, so 1 is the first row 
    atlas = np.load(base_directory / 'annotation_volume_10um_by_index.npy', mmap_mode='r')
    structure_tree = pd.read_csv(base_directory / 'structure_tree_safe_2017.csv')
    return atlas, structure_tree

atlas, structure_tree = load_ccf_data(Path('/mnt/Magill_Lab/Julien/ASAP/software/allenccf'))


def get_region_boundaries(coords, atlas, structure_tree):
    # the trajectory is from the surface fo the brain to the base of the brain
    # the end point is not the ending of the probe. Neuropixel probe is about 4000um long if only the tip is used
    trajectory_depth = np.linalg.norm(np.diff(coords, axis=0)) * 10 # axis unit is in 10um, so we convert it back to 1um
    n_coords = int(trajectory_depth) # sample every 1um
    trajector_depth_list = np.linspace(0, trajectory_depth, n_coords)

    coords_sampled = np.zeros((n_coords, 3)).astype(int)
    for i in range(3):
        coords_sampled[:, i] = np.linspace(coords[0, i], coords[1, i], n_coords).astype(int)

    # get the annotation index from the coordinates
    annot = np.zeros((coords_sampled.shape[0],))
    for i in range(coords_sampled.shape[0]):
        annot[i] = atlas[coords_sampled[i, 0], coords_sampled[i, 1], coords_sampled[i, 2]] - 1  # convert it to zero based

    # Find out the boundary regions
    # Find the boundary of the region, where the annot index changes
    regions_bins = [0, *(np.nonzero(np.diff(annot) != 0)[0] + 1), annot.shape[0]-1]  # idx of annot that denotes the boundaries
    region_boundaries = [regions_bins[:-2], regions_bins[1:-1]]  # the beginning and end of the region

    trajectory_areas = structure_tree.loc[annot[region_boundaries[0]]]
    trajectory_areas['depth_start'] = trajector_depth_list[region_boundaries[0]]
    trajectory_areas['depth_end'] = trajector_depth_list[region_boundaries[1]]
    # Note: the actual may be a few microns different from the AP_historylogy results due to
    # slightly different sampling point for the trajectory 
    
    return trajectory_areas

def shift_trajectory_depth(coords, shift_depth,axis_resolution=10):
    # Calculate the direction vector
    # coords is [start, end] in axis coordinate
    # shift_depth is in um, positive is deeper into the brain
    direction_vector = coords[1] - coords[0]
    
    # Normalize the direction vector
    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
    
    # Apply the shift
    coords_shifted = np.zeros_like(coords)
    coords_shifted[0] = coords[0] + shift_depth/axis_resolution * direction_vector_normalized
    coords_shifted[1] = coords[1] + shift_depth/axis_resolution * direction_vector_normalized
    
    return coords_shifted

# Example usage
shifted_coords = shift_trajectory_depth(coords, -1000)
trajectory_areas = get_region_boundaries(shifted_coords, atlas, structure_tree)
trajectory_areas[['acronym','depth_start','depth_end']]
# %%
