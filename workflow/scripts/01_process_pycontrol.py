
#%%
import pandas as pd
import seaborn as sns
import numpy as np
from trialexp.process.pycontrol.session_analysis import *
from trialexp.process.pycontrol.utils import *
from trialexp.process.pycontrol.plot_utils import *
from trialexp.process.pycontrol.session_analysis import Session # do not look used
from trialexp.process.pycontrol.data_import import session_dataframe
from datetime import datetime
from snakehelper.SnakeIOHelper import getSnake
from pathlib import Path
from trialexp import config
from trialexp.process.processor.BasePycontrolProcessor import BasePycontrolProcessor
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/df_events_cond.pkl'],
  'process_pycontrol')

#%% Read pycontrol file
path = Path(sinput.session_path, 'pycontrol')
filename = list(path.glob('*.txt')) + list(path.glob('*.tsv'))
if len(filename)>1:
    raise ValueError('There are more than one pycontrol file there', filename)


processor = BasePycontrolProcessor()
df_pycontrol = processor.load_pycontrol(filename[0], sinput.session_path)

#%% process lick events

df_pycontrol = processor.process_lick_events(df_pycontrol, path)

    
df_pycontrol.to_pickle(soutput.pycontrol_dataframe)

    
#%% Read task definition
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)

df_conditions, df_events_cond, df_events_trials = processor.process_trial_info(df_pycontrol, tasks)
# %% save

df_events_cond.to_pickle(soutput.event_dataframe)
df_conditions.to_pickle(soutput.condition_dataframe)
df_events_trials.to_pickle(soutput.trial_dataframe)


# %%
