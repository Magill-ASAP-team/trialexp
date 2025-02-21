# Triexap

This repository contains the analysis pipeline for pycontrol, pyphotometry and Neuropixels data.

## Installation
1. Create a virtual environment in conda.  
`conda create -n trialexp python=3.9`
2. Install the dependencies  
`pip install -r requirements.txt`
3. Install this package  
`pip install -e .`

## Usage
For detailed usage, please consult the documentation inside the `workflow` folder.


## Adding new cohort
Files to modify:
1. `params/animal_info.csv`
2. `workflow/scripts/00_create_session_folders.py`
3. `workflow/pycontrol.smk`

## Adding new task
Files to modify
1.  `params/tasks_params.csv`
2. the `compute_success` function in `trialexp/process/pycontrol/session_analysis.py`
3. `workflow/scripts/07_time_warping.py`
4. the `compute_trial_outcome` function in `trialexp/process/pycontrol/session_analysis.py`