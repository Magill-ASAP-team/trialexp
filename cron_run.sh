#!/bin/bash
cd /home/MRC.OX.AC.UK/ndcn1330/code/tmp/trialexp
source .venv/bin/activate
git pull
uv sync
just 