# %%
import os
import dotenv
from trialexp.process.pycontrol.utils import auto_load_dotenv
from pathlib import Path


# make sure we load the .env file from the correct location
cwd = Path(os.getcwd())
if cwd.name == 'scripts':
    auto_load_dotenv(str(cwd.parents[1]))
else:
    auto_load_dotenv()

# debug_folder = r'//ettin/Magill_Lab/Teris/ASAP/expt_sessions/kms063-2023-02-27-164426/'
# debug_folder = r'//home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/_Other/test_folder_ephys/kms058_2023-03-24_15-09-44_bar_final/Record Node 101/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA-AP/sorting/'
debug_folder = os.path.join(os.environ['SESSION_ROOT_DIR'], 
                            '2024_August_cohort',
                            'by_sessions',
                            'reaching_go_spout_bar_VR_April24',
                            'RE015-2024-07-17-174310')
# debug_folder = r'/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/by_sessions/reaching_go_spout_bar_nov22/kms062-2023-03-06-182344'
# %%
