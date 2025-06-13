#!/bin/bash
source .venv/bin/activate
cd /home/MRC.OX.AC.UK/ndcn1330/code/tmp/trialexp
git pull
uv sync
just 