from trialexp.process.tasks.base import TaskAnalysis
import pandas as pd
import xarray as xr
from trialexp.process.pycontrol.session_analysis import *
from trialexp.process.pycontrol.utils import *
from trialexp.process.pycontrol.plot_utils import *
from trialexp.process.pycontrol.session_analysis import Session # do not look used
from trialexp.process.pycontrol.data_import import session_dataframe
from trialexp.process.pycontrol import event_filters
from datetime import datetime
from pathlib import Path

class DefaultTask(TaskAnalysis):

    def process_pycontrol(self, df_pycontrol: pd.DataFrame, session_time: datetime, subjectID: str) -> pd.DataFrame:
       
        # Read task definition
        tasks = pd.read_csv('params/tasks_params.csv', index_col=0)
        task_name = df_pycontrol.attrs['task_name']
        timelim = [1000, 4000] # in ms

        (conditions, triggers, events_to_process, 
            trial_window, extra_trigger_events, trial_parameters) = get_task_specs(tasks,  task_name)

        # Extract trial-related information from events
        df_pycontrol = df_pycontrol[~(df_pycontrol.subtype=='sync')] #remove the sync pulse
        df_pycontrol  = print2event(df_pycontrol, conditions, trial_parameters)

        df_events_trials, df_events = extract_trial_by_trigger(df_pycontrol, triggers[0], 
                                                            conditions+events_to_process+triggers+trial_parameters, 
                                                trial_window, subjectID, session_time)

        df_conditions = compute_conditions_by_trial(df_events_trials, conditions)
        df_conditions = add_trial_params(df_conditions, trial_parameters, df_events)
        df_conditions = compute_success(df_events_trials, df_conditions,
                                        task_name, triggers, timelim)

        df_conditions['trial_outcome'] = df_conditions.apply(compute_trial_outcome, 
                                                        task_name = df_pycontrol.attrs['task_name'],
                                                        axis=1)

        #  Merge condition back with event dataframe
        df_events_cond = df_events.merge(df_conditions, on='trial_nb')

        # Add in all the meta information
        df_events_cond.attrs.update(df_events.attrs)
        df_events_cond.attrs.update(
            {'conditions': conditions,
                'triggers': triggers,
                'extra_event_triggers': extra_trigger_events, # used to average over additional events without considering the trial structure
                'events_to_process': events_to_process}
        )

        return df_events_cond
    
    def run_behavioral_analysis(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame):
        first_reach_travel_time = df_events.groupby('trial_nb').apply(event_filters.get_reach_travel_time)
        xr_first_reach_time = xr.DataArray(first_reach_travel_time)
        
        first_sig_bar_off_time = df_events.groupby('trial_nb').apply(event_filters.get_first_sig_bar_off_time)
        xr_first_sig_bar_off_time = xr.DataArray(first_sig_bar_off_time)

        xr_behaviour = xr.Dataset({'first_reach_travel_time':xr_first_reach_time,
                                    'first_sig_bar_off_trial_time': xr_first_sig_bar_off_time})

        xr_behaviour = xr_behaviour.expand_dims({'session_id':[df_events.attrs['session_id']]})

        return xr_behaviour

    def plot_results(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame):
        # Not implemented for this task
        return super().plot_results(df_events, df_conditions)
