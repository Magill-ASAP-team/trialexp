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
    # recording_csv = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/ephys/states.npy')
    # photometry = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/pyphotometry/*.ppd')
    # spike_workflow = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/spike_workflow.done')
    xr_timewarpped = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/xr_spikes_timewarped.nc')
    xr_photom_timewarped = glob(f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/xr_photom_timewarped.nc')

    if len(xr_timewarpped) > 0 and len(xr_photom_timewarped) > 0:
        return f'{wildcards.sessions}/{wildcards.task_path}/{wildcards.session_id}/processed/modelling.done'
    else:
        return []


def session2analyze(tasks:list=None, cohort:list = None, animals:list = None):
    #specify the list of task to analyze to save time.
    total_sessions = []

    if tasks is None:
        tasks=['*']

    if cohort is None:
        cohort = ['*']

    if animals is None:
        animals = ['*']
    else:
        animals = [f'{a}*' for a in animals]
    

    for c in cohort:
        for a in animals:
            for t in tasks:
                session_paths = list(Path(SESSION_ROOT_DIR).glob(f'{c}/by_sessions/{t}/{a}'))
                for session_path in session_paths:
                    target_file = f"{str(session_path)}/processed/modelling_workflow.done"
                    total_sessions.append(target_file)  
                
    return total_sessions

rule run_model_all:
    input: session2analyze()





rule compute_mutual_information:
    input:
        xr_timewarpped = '{sessions}/{task_path}/{session_id}/processed/xr_spikes_timewarped.nc',
        xr_photom_timewarped = '{sessions}/{task_path}/{session_id}/processed/xr_photom_timewarped.nc', 
    log:
        '{sessions}/{task_path}/{session_id}/processed/log/compute_mutual_information.log'  
    output:
        xr_mi = '{sessions}/{task_path}/{session_id}/processed/xr_mi.nc',
        figures_dir = directory('{sessions}/{task_path}/{session_id}/processed/figures/modelling/mutual_info'),
    threads: 32
    script:
        "scripts/modelling/04_mutual_information.py"


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


rule modelling:
    input:
        # '{sessions}/{task_path}/{session_id}/processed/modelling_cv.done',
        '{sessions}/{task_path}/{session_id}/processed/xr_mi.nc'
    output:
        touch('{sessions}/{task_path}/{session_id}/processed/modelling.done')
        

rule modelling_workflow_final:
    input:
        rec_properties_input
    output:
        done = touch('{sessions}/{task_path}/{session_id}/processed/modelling_workflow.done')

