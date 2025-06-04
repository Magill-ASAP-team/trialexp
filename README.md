# Triexap

This repository contains the analysis pipeline for pycontrol, pyphotometry and Neuropixels data.

## Installation
This project manages its dependencis via uv
1. Install `uv` following instruction [here](https://docs.astral.sh/uv/getting-started/installation/)
2. At the project directory, run `uv sync`

## Usage
This pipline uses [just](https://github.com/casey/just) to handle receipes to run the pipeline
1. Run `just run-pipeline` to execute the full pipeline
2. To force rerun a certain session use `just make-session <search term> -F`  
 e.g. `just make-session TT021-2025-06-04 -F`

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