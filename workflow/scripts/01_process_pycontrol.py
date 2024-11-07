
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
import settings
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [settings.debug_folder + '/processed/df_events_cond.pkl'],
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

if len(lick_signal_path)>0:
    lick_signal_path = list(path.glob('*_lick_analog.data*'))[0]
    lick_ts_path = list(path.glob('*_lick_analog.time*'))[0]
    lick_on, lick_off, lick = analyze_lick_signal(lick_signal_path, lick_ts_path)
    # add lick back to pycontrol
    df_pycontrol = add_lick_events(df_pycontrol, lick_on, lick_off)

    
df_pycontrol.to_pickle(soutput.pycontrol_dataframe)

    
#%% Read task definition
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)

timelim = [1000, 4000] # in ms

(conditions, triggers, events_to_process, 
    trial_window, extra_trigger_events, trial_parameters) = get_task_specs(tasks,  task_name)

#%% Extract trial-related information from events
df_pycontrol = df_pycontrol[~(df_pycontrol.subtype=='sync')] #remove the sync pulse
df_pycontrol  = print2event(df_pycontrol, conditions, trial_parameters)

df_events_trials, df_events = extract_trial_by_trigger(df_pycontrol, triggers[0], 
                                                       conditions+events_to_process+triggers+trial_parameters, 
                                            trial_window, subjectID, session_time)

df_conditions = compute_conditions_by_trial(df_events_trials, conditions)
df_conditions = add_trial_params(df_conditions, trial_parameters, df_events)
df_conditions = compute_success(df_events_trials, df_conditions,
                                  task_name, triggers, timelim)


#%%

df_conditions['trial_outcome'] = df_conditions.apply(compute_trial_outcome, 
                                                   task_name = df_pycontrol.attrs['task_name'],
                                                   axis=1)


#%%  Merge condition back with event dataframe

df_events_cond = df_events.merge(df_conditions, on='trial_nb')

#%% Add in all the meta information

df_events_cond.attrs.update(df_events.attrs)
df_events_cond.attrs.update(
    {'conditions': conditions,
     'triggers': triggers,
     'extra_event_triggers': extra_trigger_events, # used to average over additional events without considering the trial structure
     'events_to_process': events_to_process}
)

# %% save

df_events_cond.to_pickle(soutput.event_dataframe)
df_conditions.to_pickle(soutput.condition_dataframe)
df_events_trials.to_pickle(soutput.trial_dataframe)


# %%

def get_session_info(df_pycontrol, df_conditions):
    df_info = pd.DataFrame(df_pycontrol.attrs.items(), columns=['info', 'value'])
    
    start_params = df_pycontrol[df_pycontrol.subtype=='run_start'].iloc[0].content
    start_params = {k:v for k,v in start_params.items() if not k.endswith('___')}
    df_start_params = pd.DataFrame(start_params.items(), columns=['info', 'value'])

    user_param = df_pycontrol[df_pycontrol.subtype=='user_set'].content
    user_param = [item for row in user_param for item in row.items()]
    df_user_param = pd.DataFrame(user_param, columns=['variable','value'])

    df_conditions.trial_outcome.value_counts()/len(df_conditions)
    df_trial_outcome_counts = df_conditions.trial_outcome.value_counts(normalize=True).reset_index()
    df_trial_outcome_counts.columns = ['trial_outcome', 'proportion']
    df_trial_outcome_counts
    
    return df_info, df_start_params, df_user_param, df_trial_outcome_counts

df_info, df_start_params, df_user_param, df_trial_outcome_counts = get_session_info(df_pycontrol, df_conditions)
