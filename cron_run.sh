#!/bin/bash
echo "========Running pipeline==========="
echo $(date)
source /home/MRC.OX.AC.UK/ndcn1330/miniconda3/etc/profile.d/conda.sh
conda activate trialexp2
cd /home/MRC.OX.AC.UK/ndcn1330/code/tmp/
git clone https://github.com/Magill-ASAP-team/trialexp.git 
conda activate trialexp2
cd trialexp
python workflow/scripts/00_create_session_folders.py
snakemake --snakefile workflow/pycontrol.smk -k -c20 --rerun-triggers mtime -q progress
mv /home/MRC.OX.AC.UK/ndcn1330/code/tmp/trialexp/.snakemake /home/MRC.OX.AC.UK/ndcn1330/code/trialexp/.snakemake
rm -rf /home/MRC.OX.AC.UK/ndcn1330/code/tmp/trialexp