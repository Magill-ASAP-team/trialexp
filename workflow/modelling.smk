from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv
from trialexp.config import SESSION_ROOT_DIR

load_dotenv()

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/states.npy')
    if len(recording_csv) > 0:
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/modelling.done'
    else:
        return []

rule train_sparse_model:
    input:
        xr_timewarpped = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
        xr_photom_timewarped = '{sessions}/{task_path}/{session_id}/processed/xr_photom_timewarped.nc', 
    output:
        ach_model = '{sessions}/{task_path}/{session_id}/processed/ach_sparse_encode.pkl',
        da_model = '{sessions}/{task_path}/{session_id}/processed/da_sparse_encode.pkl',
        done = '{sessions}/{task_path}/{session_id}/processed/modelling.done'
    script:
        "scripts/modelling/01_train_sparse_model.py"

rule modelling_workflow_final:
    input:
        rec_properties_input
    output:
        done = touch('{sessions}/{task_path}/{session_id}/processed/modelling_workflow.done')
