#!/bin/bash
source /home/MRC.OX.AC.UK/ndcn1330/miniconda3/etc/profile.d/conda.sh
conda activate trialexp2
cd /home/MRC.OX.AC.UK/ndcn1330/code/tmp/trialexp
git pull 
conda activate trialexp2
python workflow/scripts/00_create_session_folders.py
snakemake --snakefile workflow/pycontrol.smk -k
