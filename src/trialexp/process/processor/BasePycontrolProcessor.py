from trialexp.process.pycontrol.data_import import session_dataframe
from trialexp.process.pycontrol.session_analysis import *
from trialexp.process.pycontrol.utils import *

class BasePycontrolProcessor:
    def load_pycontrol(self, filename:str, session_path:str):
        
        df_session = session_dataframe(filename)
        df_pycontrol = parse_session_dataframe(df_session)
        # if df_pycontrol.attrs['framework_version'] in ['1.8.1','1.8']:
        #     session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S')
        # else:
        #     session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%f')


        # subjectID = df_pycontrol.attrs['subject_id']
        # task_name = df_pycontrol.attrs['task_name']
        session_id = Path(session_path).name

        df_pycontrol.attrs['session_id'] = session_id
        
        return df_pycontrol
    
    def process_lick_events(self, df_pycontrol, path):
        """Process lick events from analog signal if not enough lick events detected."""
        lick_signal_path = list(path.glob('*_lick_analog.data*'))
        lick_events = df_pycontrol[df_pycontrol.content=='lick']

        if len(lick_events) < 10 and len(lick_signal_path) > 0:
            # only do it if there is no lick detected using other means
            lick_signal_path = list(path.glob('*_lick_analog.data*'))[0]
            lick_ts_path = list(path.glob('*_lick_analog.time*'))[0]
            lick_on, lick_off, lick = analyze_lick_signal(lick_signal_path, lick_ts_path)
            # add lick back to pycontrol
            df_pycontrol = add_lick_events(df_pycontrol, lick_on, lick_off)
        
        return df_pycontrol
    
    def process_trial_info(self, df_pycontrol:pd.DataFrame, tasks:dict):
        timelim = [1000, 4000] # in ms

        # get information from the pycontrol file
        subjectID = df_pycontrol.attrs['subject_id']
        task_name = df_pycontrol.attrs['task_name']
        if df_pycontrol.attrs['framework_version'] in ['1.8.1','1.8']:
            session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S')
        else:
            session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%f')

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
        
        return df_conditions, df_events_cond, df_events_trials