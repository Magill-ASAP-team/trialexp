#!/bin/bash
cd /home/MRC.OX.AC.UK/ndcn1330/code/trialexp
export PATH="/home/MRC.OX.AC.UK/ndcn1330/.local/bin/:$PATH" #manually add uv to path
git pull
uv sync
just 