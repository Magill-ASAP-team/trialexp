#!/bin/bash
cd /home/MRC.OX.AC.UK/ndcn1330/code/tmp/trialexp
git pull
/home/MRC.OX.AC.UK/ndcn1330/.local/bin/uv sync
/home/MRC.OX.AC.UK/ndcn1330/code/trialexp/.venv/bin/just 