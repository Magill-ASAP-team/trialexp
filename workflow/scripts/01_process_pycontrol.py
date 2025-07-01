
#%%
import pandas as pd
import seaborn as sns
import numpy as np
from trialexp.process.tasks.factory import get_task_analysis
from snakehelper.SnakeIOHelper import getSnake
from pathlib import Path
from trialexp import config
from trialexp.process.pycontrol.data_import import session_dataframe
from datetime import datetime
from trialexp.process.pycontrol.utils import analyze_lick_signal, add_lick_events, parse_session_dataframe
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/df_events_cond.pkl'],
  'process_pycontrol')

#%% Read pycontrol file
path = Path(sinput.session_path, 'pycontrol')
filename = list(path.glob('*.txt')) + list(path.glob('*.tsv'))
if len(filename)>1:
    raise ValueError('There are more than one pycontrol file there', filename)

df_session = session_dataframe(filename[0])
df_pycontrol = parse_session_dataframe(df_session)
if df_pycontrol.attrs['framework_version'] in ['1.8.1','1.8']:
    session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S')
else:
    session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%f')


subjectID = df_pycontrol.attrs['subject_id']
task_name = df_pycontrol.attrs['task_name']
session_id = Path(sinput.session_path).name

df_pycontrol.attrs['session_id'] = session_id

#%% process lick events
# check if lick event is in the pycontrol file, if not, try to analyze it from the analog signal
lick_signal_path = list(path.glob('*_lick_analog.data*'))
lick_events = df_pycontrol[df_pycontrol.content=='lick']

if len(lick_events)<10 and len(lick_signal_path)>0:
    # only do it if there is no lick detected using other means
    lick_signal_path = list(path.glob('*_lick_analog.data*'))[0]
    lick_ts_path = list(path.glob('*_lick_analog.time*'))[0]
    lick_on, lick_off, lick = analyze_lick_signal(lick_signal_path, lick_ts_path)
    # add lick back to pycontrol
    df_pycontrol = add_lick_events(df_pycontrol, lick_on, lick_off)

    
df_pycontrol.to_pickle(soutput.pycontrol_dataframe)


# %% Get task analysis class

task_analysis = get_task_analysis(task_name, session_path=sinput.session_path)
df_events_cond = task_analysis.process_pycontrol(df_pycontrol, session_time, subjectID)

# %% save

df_events_cond.to_pickle(soutput.event_dataframe)




