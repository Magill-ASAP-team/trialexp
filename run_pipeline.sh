#!/bin/bash

# Activate the conda environment
source /home/MRC.OX.AC.UK/ndcn1330/miniconda3/etc/profile.d/conda.sh
conda activate trialexp2

# Run the Python file
python workflow/scripts/00_create_session_folders.py

snakemake --snakefile workflow/pycontrol.smk -k -c20 --rerun-triggers mtime
