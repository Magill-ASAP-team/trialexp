from glob import glob
from pathlib import Path
import os 
from dotenv import load_dotenv
from trialexp.config import SESSION_ROOT_DIR

load_dotenv()

def check_photometry_exists(wildcards):
    ppd_files = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/pyphotometry/*.ppd')
    if len(ppd_files) > 0:
        return True
    else:
        return False

def rec_properties_input(wildcards):
    # determine if there is an ephys recording for that folder
    recording_csv = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/states.npy')
    photometry = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/pyphotometry/*.ppd')

    if len(recording_csv) > 0 and len(photometry) > 0:
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/modelling.done'
    else:
        return []

rule train_sparse_model:
    input:
        xr_timewarpped = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
        xr_photom_timewarped = '{sessions}/{task_path}/{session_id}/processed/xr_photom_timewarped.nc', 
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/train_sparse_model.log'  
    output:
        ach_model = '{sessions}/{task_path}/{session_id}/processed/ach_sparse_encode.pkl',
        da_model = '{sessions}/{task_path}/{session_id}/processed/da_sparse_encode.pkl',
        figures_dir = directory('{sessions}/{task_path}/{session_id}/processed/figures/modelling'),
        done = touch('{sessions}/{task_path}/{session_id}/processed/modelling.done')
    threads: 32
    script:
        "scripts/modelling/01_train_sparse_model.py"

rule shuffle_sparse_model:
    input:
        xr_timewarpped = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
        xr_photom_timewarped = '{sessions}/{task_path}/{session_id}/processed/xr_photom_timewarped.nc',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/train_sparse_model.log'
    output:
        ach_model = '{sessions}/{task_path}/{session_id}/processed/ach_sparse_encode_shuffle.pkl',
        da_model = '{sessions}/{task_path}/{session_id}/processed/da_sparse_encode_shuffle.pkl',
        # done = touch('{sessions}/{task_path}/{session_id}/processed/modelling.done')
    threads: 32
    script:
        "scripts/modelling/02_sparse_model_shuffle.py"

rule train_sparse_model_cv:
    input:
        xr_timewarpped = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
        xr_photom_timewarped = '{sessions}/{task_path}/{session_id}/processed/xr_photom_timewarped.nc',
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/train_sparse_model_cv.log'
    output:
        ach_model = '{sessions}/{task_path}/{session_id}/processed/ach_sparse_encode_cv.pkl',
        da_model = '{sessions}/{task_path}/{session_id}/processed/da_sparse_encode_cv.pkl',
        figures_dir = directory('{sessions}/{task_path}/{session_id}/processed/figures/modelling_cv'),
        done = touch('{sessions}/{task_path}/{session_id}/processed/modelling_cv.done')
    threads: 32
    script:
        "scripts/modelling/03_train_sparse_model_cv.py"

rule modelling_workflow_final:
    input:
        rec_properties_input
    output:
        done = touch('{sessions}/{task_path}/{session_id}/processed/modelling_workflow.done')
