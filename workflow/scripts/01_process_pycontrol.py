
#%%
import pandas as pd
import seaborn as sns
import numpy as np
from trialexp.process.pycontrol.session_analysis import *
from trialexp.process.pycontrol.utils import *
from trialexp.process.pycontrol.plot_utils import *
from trialexp.process.pycontrol.session_analysis import Session # do not look used
from trialexp.process.pycontrol.processors import get_processor
from trialexp.process.pycontrol.data_import import session_dataframe
from datetime import datetime
from snakehelper.SnakeIOHelper import getSnake
from pathlib import Path
from trialexp import config
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/df_events_cond.pkl'],
  'process_pycontrol')

#%% Load task configuration and get processor
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)

# Get processor configuration - read task name from a dummy parse first
path = Path(sinput.session_path, 'pycontrol')
filename = list(path.glob('*.txt')) + list(path.glob('*.tsv'))
if len(filename)>1:
    raise ValueError('There are more than one pycontrol file there', filename)
if len(filename)==0:
    raise ValueError('No pycontrol files found in', path)
    
# Quick parse to get task name for processor selection
df_session = session_dataframe(filename[0])
df_pycontrol = parse_session_dataframe(df_session)

session_id = Path(sinput.session_path).name
task_name = df_pycontrol.attrs['task_name']

# Get processor configuration
task_idx = (tasks["task"] == task_name)
processor_class_name = str(tasks.loc[task_idx, 'processor_class'].iloc[0]) if 'processor_class' in tasks.columns else 'BaseTaskProcessor'
processor = get_processor(processor_class_name)

#%% Process full session using processor

task_config = processor.get_task_configuration(tasks, task_name)
df_events_cond, df_conditions, df_events_trials = processor.process_full_session(
    df_pycontrol, task_config
)

# Save pycontrol dataframe (first save point)
df_pycontrol.to_pickle(soutput.pycontrol_dataframe)

# %% Save all processed data
df_events_cond.to_pickle(soutput.event_dataframe)
df_conditions.to_pickle(soutput.condition_dataframe)
df_events_trials.to_pickle(soutput.trial_dataframe)


# %%
