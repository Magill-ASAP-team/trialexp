"""
PyControl processor for core behavioral data processing functionality.

This processor handles PyControl session parsing, trial extraction, condition computation,
and task-specific success criteria.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
from trialexp.process.pycontrol.utils import (
    find_if_event_within_timelim, 
    find_last_time_before_list,
    parse_session_dataframe,
    analyze_lick_signal,
    add_lick_events,
    print2event
)
from trialexp.process.pycontrol.session_analysis import (
    get_task_specs,
    extract_trial_by_trigger,
    compute_conditions_by_trial,
    add_trial_params
)
from trialexp.process.pycontrol.data_import import session_dataframe
from trialexp.process.pycontrol.plot_utils import (
    plot_event_distribution, 
    style_event_distribution, 
    reach_time
)
from trialexp.process.pycontrol.utils import (
    get_windowed_discriminability_score,
    discrminability_before_after
)
import matplotlib.pyplot as plt


class PyControlProcessor:
    """
    Processor for core PyControl behavioral data functionality.
    
    Handles session parsing, trial extraction, condition computation,
    and task-specific success criteria without photometry dependencies.
    """
    
    def load_and_parse_pycontrol(self, session_path: str, task_config: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Load and parse PyControl session file.
        
        Args:
            session_path: Path to PyControl session file
            tasks_df: Tasks configuration dataframe
            
        Returns:
            Tuple of (session_dataframe, task_name)
        """
        session_name = Path(session_path).stem
        df_session = session_dataframe(session_path)
        
        # Add metadata
        df_session.attrs['session_name'] = session_name
        df_session.attrs['session_path'] = session_path
        df_session.attrs['loading_time'] = datetime.now().isoformat()
        
        # Determine task name
        task_name = self.get_task_name(df_session, tasks_df)
        df_session.attrs['task_name'] = task_name
        
        return df_session, task_name
    
    def get_task_name(self, df_session: pd.DataFrame, tasks_df: pd.DataFrame) -> str:
        """
        Determine task name from session data and configuration.
        
        Args:
            df_session: Session dataframe
            tasks_df: Tasks configuration dataframe
            
        Returns:
            str: Task name
        """
        # Try to get task name from session metadata
        if hasattr(df_session, 'attrs') and 'task_name' in df_session.attrs:
            return df_session.attrs['task_name']
        
        # Fallback to first task in configuration
        if len(tasks_df) > 0:
            return tasks_df.iloc[0]['task']
        
        return 'unknown_task'
    
    def get_task_configuration(self, tasks_df: pd.DataFrame, task_name: str) -> Dict[str, Any]:
        """
        Get task-specific configuration parameters.
        
        Args:
            task_name: Name of the task
            tasks_df: Tasks configuration dataframe
            
        Returns:
            Dict containing task configuration
        """
        task_specs = get_task_specs(tasks_df, task_name)
        # Convert tuple from get_task_specs to dictionary
        task_config = {
            'conditions': task_specs[0],
            'triggers': task_specs[1],
            'events_to_process': task_specs[2],
            'trial_window': task_specs[3],
            'extra_event_trigger': task_specs[4],
            'trial_parameters': task_specs[5]
        }
        return task_config
    
    def extract_trial_data(self, df_session: pd.DataFrame, task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract trial-based events from session data.
        
        Args:
            df_session: Session dataframe
            task_config: Task configuration dictionary
            
        Returns:
            pd.DataFrame: Trial events dataframe
        """
        triggers = task_config.get('triggers', ['hold_for_water'])
        trial_window = task_config.get('trial_window', [-1000, 4000])
        events_to_process = task_config.get('events_to_process', [])
        
        # Get session metadata
        subject_ID = df_session.attrs.get('subject_ID', 'unknown')
        datetime_obj = df_session.attrs.get('datetime_obj', None)
        
        # Provide fallback datetime if not available
        if datetime_obj is None:
            from datetime import datetime
            datetime_obj = datetime.now()
        
        # Extract trials using the first trigger
        df_events_trials, df_events = extract_trial_by_trigger(df_session, triggers[0], events_to_process, trial_window, subject_ID, datetime_obj)
        
        # Add metadata
        df_events.attrs.update(df_session.attrs)
        df_events.attrs['triggers'] = triggers
        df_events.attrs['trial_window'] = trial_window
        
        # Store df_events_trials for use in conditions computation
        df_events.attrs['df_events_trials'] = df_events_trials
        
        return df_events
    
    def compute_trial_conditions(self, df_events: pd.DataFrame, task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute trial conditions and parameters.
        
        Args:
            df_events: Trial events dataframe
            task_config: Task configuration dictionary
            
        Returns:
            pd.DataFrame: Conditions dataframe
        """
        # Get conditions and df_events_trials from task config and stored attributes
        conditions = task_config.get('conditions', [])
        trial_parameters = task_config.get('trial_parameters', [])
        df_events_trials = df_events.attrs.get('df_events_trials')
        
        # Compute basic conditions
        df_conditions = compute_conditions_by_trial(df_events_trials, conditions)
        
        # Add trial parameters if available
        df_conditions = add_trial_params(df_conditions, trial_parameters, df_events)
        
        # Add metadata
        df_conditions.attrs.update(df_events.attrs)
        
        return df_conditions
    
    def compute_trial_outcome(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                            task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute trial outcomes based on task-specific criteria.
        
        Args:
            df_events: Trial events dataframe
            df_conditions: Conditions dataframe
            task_config: Task configuration dictionary
            
        Returns:
            pd.DataFrame: Conditions dataframe with trial outcomes
        """
        df_conditions = df_conditions.copy()
        
        # Initialize outcome column
        df_conditions['trial_outcome'] = 'unknown'
        
        task_name = task_config.get('task_name', '')
        triggers = task_config.get('triggers', [])
        timelim = task_config.get('timelim', [1000, 4000])
        
        # Task-specific outcome computation
        if 'go_nogo' in task_name.lower():
            df_conditions = self._compute_go_nogo_outcomes(df_events, df_conditions, triggers, timelim)
        elif 'pavlovian' in task_name.lower():
            df_conditions = self._compute_pavlovian_outcomes(df_events, df_conditions, triggers, timelim)
        elif 'reaching' in task_name.lower():
            df_conditions = self._compute_reaching_outcomes(df_events, df_conditions, triggers, timelim)
        else:
            # Default outcome computation
            df_conditions = self._compute_default_outcomes(df_events, df_conditions, triggers, timelim)
        
        return df_conditions
    
    def _compute_go_nogo_outcomes(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                                triggers: List[str], timelim: List[int]) -> pd.DataFrame:
        """Compute outcomes for Go-NoGo tasks."""
        # Go trials: success if spout within timelim
        go_trials = df_conditions['trigger'] == triggers[0] if len(triggers) > 0 else pd.Series([True] * len(df_conditions))
        
        for idx in df_conditions[go_trials].index:
            if idx in df_events.index:
                spout_times = df_events.loc[idx, 'spout_trial_time'] if 'spout_trial_time' in df_events.columns else []
                has_spout = find_if_event_within_timelim(spout_times, timelim) if spout_times is not None else False
                df_conditions.loc[idx, 'trial_outcome'] = 'success' if has_spout else 'no_reach'
        
        # NoGo trials: success if no bar_off within timelim
        if len(triggers) > 1:
            nogo_trials = df_conditions['trigger'] == triggers[1]
            for idx in df_conditions[nogo_trials].index:
                if idx in df_events.index:
                    bar_off_times = df_events.loc[idx, 'bar_off_trial_time'] if 'bar_off_trial_time' in df_events.columns else []
                    has_bar_off = find_if_event_within_timelim(bar_off_times, timelim) if bar_off_times is not None else False
                    df_conditions.loc[idx, 'trial_outcome'] = 'success' if not has_bar_off else 'early_reach'
        
        return df_conditions
    
    def _compute_pavlovian_outcomes(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                                  triggers: List[str], timelim: List[int]) -> pd.DataFrame:
        """Compute outcomes for Pavlovian tasks."""
        for idx in df_conditions.index:
            if idx in df_events.index:
                spout_times = df_events.loc[idx, 'spout_trial_time'] if 'spout_trial_time' in df_events.columns else []
                has_spout = find_if_event_within_timelim(spout_times, timelim) if spout_times is not None else False
                df_conditions.loc[idx, 'trial_outcome'] = 'success' if has_spout else 'no_reach'
        
        return df_conditions
    
    def _compute_reaching_outcomes(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                                 triggers: List[str], timelim: List[int]) -> pd.DataFrame:
        """Compute outcomes for reaching tasks."""
        for idx in df_conditions.index:
            if idx in df_events.index:
                # Check for spout events
                spout_times = df_events.loc[idx, 'spout_trial_time'] if 'spout_trial_time' in df_events.columns else []
                has_spout = find_if_event_within_timelim(spout_times, timelim) if spout_times is not None else False
                
                if has_spout:
                    # Check for reward delivery or other success criteria
                    reward_events = ['US_delay_timer', 'US_jackpot', 'water_delivered']
                    has_reward = False
                    for reward_event in reward_events:
                        if f'{reward_event}_trial_time' in df_events.columns:
                            reward_times = df_events.loc[idx, f'{reward_event}_trial_time']
                            if reward_times is not None and len(reward_times) > 0:
                                has_reward = True
                                break
                    
                    df_conditions.loc[idx, 'trial_outcome'] = 'success' if has_reward else 'aborted'
                else:
                    # Check if this was a late reach
                    extended_timelim = [timelim[0], timelim[1] + 2000]  # Add 2s window
                    late_spout = find_if_event_within_timelim(spout_times, extended_timelim) if spout_times is not None else False
                    df_conditions.loc[idx, 'trial_outcome'] = 'late_reach' if late_spout else 'no_reach'
        
        return df_conditions
    
    def _compute_default_outcomes(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                                triggers: List[str], timelim: List[int]) -> pd.DataFrame:
        """Compute default outcomes based on spout events."""
        for idx in df_conditions.index:
            if idx in df_events.index:
                spout_times = df_events.loc[idx, 'spout_trial_time'] if 'spout_trial_time' in df_events.columns else []
                has_spout = find_if_event_within_timelim(spout_times, timelim) if spout_times is not None else False
                df_conditions.loc[idx, 'trial_outcome'] = 'success' if has_spout else 'no_reach'
        
        return df_conditions
    
    def compute_success(self, df_events_trials: pd.DataFrame, df_conditions: pd.DataFrame,
                       task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute success for each trial based on task-specific criteria.
        
        Args:
            df_events_trials: Trial events dataframe
            df_conditions: Conditions dataframe
            task_config: Task configuration dictionary
            
        Returns:
            pd.DataFrame: Conditions dataframe with 'success' column added
        """
        df_conditions = df_conditions.copy()
        df_conditions["success"] = False
        
        # Mark successful trials based on trial_outcome
        if 'trial_outcome' in df_conditions.columns:
            df_conditions.loc[df_conditions['trial_outcome'] == 'success', 'success'] = True
        
        return df_conditions
    
    def process_lick_events(self, df_session: pd.DataFrame) -> pd.DataFrame:
        """
        Process lick events from session data.
        
        Args:
            df_session: Session dataframe
            
        Returns:
            pd.DataFrame: Session dataframe with processed lick events
        """
        # TODO: Implement lick event processing if needed
        # Current implementation is a placeholder that returns data unchanged
        return df_session
    
    def merge_events_with_conditions(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame) -> pd.DataFrame:
        """
        Merge events dataframe with conditions.
        
        Args:
            df_events: Events dataframe
            df_conditions: Conditions dataframe
            
        Returns:
            pd.DataFrame: Merged events dataframe with conditions
        """
        # Merge conditions into events dataframe
        df_events_cond = df_events.copy()
        
        # Add condition columns to events
        for col in df_conditions.columns:
            if col not in df_events_cond.columns:
                df_events_cond[col] = df_conditions[col]
        
        # Preserve metadata
        df_events_cond.attrs.update(df_events.attrs)
        
        return df_events_cond
    
    def plot_event_distribution(self, df_events: pd.DataFrame) -> plt.Figure:
        """
        Generate event distribution plot.
        
        Args:
            df_events: Events dataframe
            
        Returns:
            matplotlib Figure: Event distribution plot
        """
        try:
            fig = plot_event_distribution(df_events)
            style_event_distribution(fig)
            return fig
        except Exception as e:
            print(f"Warning: Could not generate event distribution plot: {e}")
            # Return empty figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Event distribution plot failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Event Distribution (Error)')
            return fig