"""
Base task processor containing the default processing logic for PyControl behavioral tasks.

This processor contains the core logic moved from session_analysis.py and serves as
the foundation for task-specific processors.
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
import seaborn as sns
import xarray as xr
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.utils import add_events_to_time_series, calculate_lick_rate
from trialexp.process.pyphotometry.utils import (
    import_ppd_auto, 
    preprocess_photometry, 
    photometry2xarray, 
    add_event_data,
    align_photometry_to_pycontrol
)
from trialexp.utils.pyphotometry_utilities import create_photo_sync
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.event_filters import extract_event_time
import pickle
import json
import trialexp.process.pyphotometry.linear_modelling as lm


class BaseTaskProcessor:
    """
    Base processor for PyControl behavioral task analysis.
    
    Contains default implementations of success computation and trial outcome logic
    that work for most tasks. Task-specific processors can inherit from this class
    and override only the methods that need customization.
    """
    
    def process_full_session(self, session_path: str, tasks_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete processing pipeline from raw PyControl files to final results.
        
        This method handles the entire processing workflow from file loading to final
        data preparation. Scripts are responsible for saving the returned data.
        
        Args:
            session_path: Path to the session directory containing pycontrol folder
            tasks_df: DataFrame containing task configuration from tasks_params.csv
            
        Returns:
            Tuple of (df_pycontrol, df_events_cond, df_conditions, df_events_trials)
        """
        # Parse PyControl files
        df_pycontrol = self.load_and_parse_pycontrol(session_path)
        
        # Get task configuration
        task_name = df_pycontrol.attrs['task_name']
        task_specs = self.get_task_configuration(tasks_df, task_name)
        
        # Process lick events if needed
        df_pycontrol = self.process_lick_events(df_pycontrol, session_path)
        
        # Extract trial-based data
        df_events_trials, df_events = self.extract_trial_data(df_pycontrol, task_specs)
        
        # Compute conditions and success metrics
        df_conditions = self.compute_trial_conditions(df_events_trials, df_events, task_specs)
        
        # Merge events with conditions
        df_events_cond = self.merge_events_with_conditions(df_events, df_conditions, task_specs)
        
        return df_pycontrol, df_events_cond, df_conditions, df_events_trials
    
    def process_session(self, df_events_trials: pd.DataFrame, df_conditions: pd.DataFrame, 
                       task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Process trial conditions (success and outcome computation only).
        
        This is the legacy method for backward compatibility with existing refactored scripts.
        
        Args:
            df_events_trials: DataFrame with trial-based event data
            df_conditions: DataFrame with trial condition information
            task_config: Dictionary containing task configuration parameters
            
        Returns:
            pd.DataFrame: Processed conditions dataframe with success and outcome columns
        """
        # Apply processing steps in sequence
        df_conditions = self.compute_success(df_events_trials, df_conditions, task_config)
        df_conditions = self.compute_trial_outcome(df_conditions, task_config)
        
        return df_conditions
    
    def load_and_parse_pycontrol(self, session_path: str) -> pd.DataFrame:
        """
        Load and parse PyControl session files.
        
        Args:
            session_path: Path to session directory
            
        Returns:
            pd.DataFrame: Parsed PyControl dataframe with metadata
        """
        path = Path(session_path, 'pycontrol')
        filename = list(path.glob('*.txt')) + list(path.glob('*.tsv'))
        
        if len(filename) > 1:
            raise ValueError('There are more than one pycontrol file there', filename)
        if len(filename) == 0:
            raise ValueError('No pycontrol files found in', path)
            
        # Parse session data
        df_session = session_dataframe(filename[0])
        df_pycontrol = parse_session_dataframe(df_session)
        
        # Parse session time based on framework version
        if df_pycontrol.attrs['framework_version'] in ['1.8.1', '1.8']:
            session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S')
        else:
            session_time = datetime.strptime(df_pycontrol.attrs['start_time'], '%Y-%m-%dT%H:%M:%S.%f')
        
        # Add session metadata
        session_id = Path(session_path).name
        df_pycontrol.attrs['session_id'] = session_id
        df_pycontrol.attrs['parsed_session_time'] = session_time
        
        return df_pycontrol
    
    def get_task_configuration(self, tasks_df: pd.DataFrame, task_name: str) -> Dict[str, Any]:
        """
        Get task configuration from tasks dataframe.
        
        Args:
            tasks_df: Tasks configuration dataframe
            task_name: Name of the task
            
        Returns:
            Dict containing task configuration
        """
        timelim = [1000, 4000]  # Default timelim
        
        (conditions, triggers, events_to_process, 
         trial_window, extra_trigger_events, trial_parameters) = get_task_specs(tasks_df, task_name)
        
        return {
            'task_name': task_name,
            'conditions': conditions,
            'triggers': triggers, 
            'events_to_process': events_to_process,
            'trial_window': trial_window,
            'extra_trigger_events': extra_trigger_events,
            'trial_parameters': trial_parameters,
            'timelim': timelim
        }
    
    def process_lick_events(self, df_pycontrol: pd.DataFrame, session_path: str) -> pd.DataFrame:
        """
        Process lick events from analog signal if needed.
        
        Args:
            df_pycontrol: PyControl dataframe
            session_path: Path to session directory
            
        Returns:
            pd.DataFrame: PyControl dataframe with lick events added if needed
        """
        path = Path(session_path, 'pycontrol')
        lick_signal_path = list(path.glob('*_lick_analog.data*'))
        lick_events = df_pycontrol[df_pycontrol.content == 'lick']
        
        if len(lick_events) < 10 and len(lick_signal_path) > 0:
            # Only process if there are few lick events detected
            lick_signal_path = lick_signal_path[0]
            lick_ts_path = list(path.glob('*_lick_analog.time*'))[0]
            lick_on, lick_off, lick = analyze_lick_signal(lick_signal_path, lick_ts_path)
            # Add lick events back to pycontrol
            df_pycontrol = add_lick_events(df_pycontrol, lick_on, lick_off)
        
        return df_pycontrol
    
    def extract_trial_data(self, df_pycontrol: pd.DataFrame, task_specs: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract trial-based data from PyControl events.
        
        Args:
            df_pycontrol: PyControl dataframe
            task_specs: Task configuration dictionary
            
        Returns:
            Tuple of (df_events_trials, df_events)
        """
        # Remove sync pulses and process print statements
        df_pycontrol = df_pycontrol[~(df_pycontrol.subtype == 'sync')]
        df_pycontrol = print2event(df_pycontrol, task_specs['conditions'], task_specs['trial_parameters'])
        
        # Extract trial information
        subjectID = df_pycontrol.attrs['subject_id']
        session_time = df_pycontrol.attrs['parsed_session_time']
        
        df_events_trials, df_events = extract_trial_by_trigger(
            df_pycontrol, 
            task_specs['triggers'][0],
            (task_specs['conditions'] + task_specs['events_to_process'] + 
             task_specs['triggers'] + task_specs['trial_parameters']),
            task_specs['trial_window'],
            subjectID, 
            session_time
        )
        
        return df_events_trials, df_events
    
    def compute_trial_conditions(self, df_events_trials: pd.DataFrame, df_events: pd.DataFrame, 
                                task_specs: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute trial conditions, success, and outcomes.
        
        Args:
            df_events_trials: Trial-based events dataframe
            df_events: Full events dataframe
            task_specs: Task configuration dictionary
            
        Returns:
            pd.DataFrame: Conditions dataframe with success and outcome
        """
        # Compute basic conditions and trial parameters
        df_conditions = compute_conditions_by_trial(df_events_trials, task_specs['conditions'])
        df_conditions = add_trial_params(df_conditions, task_specs['trial_parameters'], df_events)
        
        # Use processor methods for success and outcome
        task_config = {
            'task_name': task_specs['task_name'],
            'triggers': task_specs['triggers'],
            'timelim': task_specs['timelim']
        }
        df_conditions = self.compute_success(df_events_trials, df_conditions, task_config)
        df_conditions = self.compute_trial_outcome(df_conditions, task_config)
        
        return df_conditions
    
    def merge_events_with_conditions(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                                   task_specs: Dict[str, Any]) -> pd.DataFrame:
        """
        Merge events dataframe with conditions and add metadata.
        
        Args:
            df_events: Events dataframe
            df_conditions: Conditions dataframe
            task_specs: Task configuration dictionary
            
        Returns:
            pd.DataFrame: Merged events and conditions dataframe
        """
        # Merge condition back with event dataframe
        df_events_cond = df_events.merge(df_conditions, on='trial_nb')
        
        # Add metadata
        df_events_cond.attrs.update(df_events.attrs)
        df_events_cond.attrs.update({
            'conditions': task_specs['conditions'],
            'triggers': task_specs['triggers'],
            'extra_event_triggers': task_specs['extra_trigger_events'],
            'events_to_process': task_specs['events_to_process']
        })
        
        return df_events_cond
    
    def generate_session_plots(self, df_events_cond: pd.DataFrame, df_pycontrol: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate standard session plots.
        
        Args:
            df_events_cond: Events dataframe with conditions
            df_pycontrol: PyControl dataframe
            
        Returns:
            Dict containing matplotlib figures for different plots
        """
        plots = {}
        
        # Event distribution plot
        plots['event_histogram'] = self.plot_event_distribution(df_events_cond)
        
        # Reach time histogram
        plots['reach_histogram'] = self.plot_reach_histogram(df_events_cond)
        
        # Task-specific plots
        task_name = df_pycontrol.attrs.get('task_name', '')
        plots['discriminability'] = self.plot_task_specific_metrics(df_pycontrol, task_name)
        
        return plots
    
    def plot_event_distribution(self, df_events_cond: pd.DataFrame) -> Any:
        """
        Plot event distribution across trials.
        
        Args:
            df_events_cond: Events dataframe with conditions
            
        Returns:
            matplotlib figure object
        """
        trial_window = df_events_cond.attrs['trial_window']
        triggers = df_events_cond.attrs['triggers']
        
        df2plot = df_events_cond.copy()
        df2plot['trial_time'] = df2plot['trial_time'] / 1000
        xlim = [trial_window[0] / 1000, np.percentile(df2plot['trial_time'], 95)]
        
        try:
            g = plot_event_distribution(df2plot, 'trial_time', 'trial_nb', 
                                      xbinwidth=1, ybinwidth=0, xlim=xlim)
            trigger_text = triggers[0].replace('_', ' ')
            style_event_distribution(g, 'Time (s)', 'Trial number', trigger_text)
        except Exception as e:
            # Fallback to simple scatter plot if main plotting fails
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, dpi=200)
            ax.scatter(df2plot['trial_time'], df2plot['trial_nb'], alpha=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial number')
            ax.set_title('Event Distribution')
            g = fig
        
        return g
    
    def plot_reach_histogram(self, df_events_cond: pd.DataFrame) -> Any:
        """
        Plot reach time histogram.
        
        Args:
            df_events_cond: Events dataframe with conditions
            
        Returns:
            matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, dpi=200)
        
        try:
            reach_t = df_events_cond.groupby('trial_nb').apply(reach_time)
            reach_t = reach_t.dropna()
            if len(reach_t) > 0:
                sns.histplot(reach_t, bins=50, binrange=(0, 500), ax=ax)
            else:
                ax.text(0.5, 0.5, 'No reach time data', ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f'Reach time analysis failed: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set(xlabel='Reach time (ms)')
        ax.set_title('Reach Time Distribution')
        
        return fig
    
    def plot_task_specific_metrics(self, df_pycontrol: pd.DataFrame, task_name: str) -> Any:
        """
        Plot task-specific metrics. Can be overridden by task-specific processors.
        
        Args:
            df_pycontrol: PyControl dataframe
            task_name: Name of the task
            
        Returns:
            matplotlib figure object or None
        """
        fig, ax = plt.subplots(1, 1, dpi=200)
        
        # Default implementation for break2 and cued_and_cued_reward tasks
        if 'break2' in task_name or 'cued_and_cued_reward' in task_name:
            df_dprime = get_windowed_discriminability_score(df_pycontrol, window_sec=3*60)
            sns.lineplot(df_dprime, x='time', y='dprime', ax=ax)
            ax.axhline(0, ls='--', color='gray')
            ax.set_title('Discriminability Score')
        else:
            # For other tasks, create empty plot
            ax.text(0.5, 0.5, 'No task-specific metrics', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Task-specific Metrics')
        
        return fig
    
    def compute_behavioral_metrics(self, df_events: pd.DataFrame) -> xr.Dataset:
        """
        Compute behavioral metrics from event data.
        
        Args:
            df_events: Events dataframe with trial information
            
        Returns:
            xarray.Dataset: Behavioral metrics dataset
        """
        # Travel time between bar off and first spout touch
        first_reach_travel_time = df_events.groupby('trial_nb').apply(event_filters.get_reach_travel_time)
        xr_first_reach_travel_time = xr.DataArray(first_reach_travel_time)
        
        # Reach time (last bar off time)
        last_bar_off_time = df_events.groupby('trial_nb').apply(event_filters.get_last_bar_off_time)
        xr_last_bar_off_time = xr.DataArray(last_bar_off_time)
        
        # Trial time of first significant bar off
        first_sig_bar_off_time = df_events.groupby('trial_nb').apply(event_filters.get_first_sig_bar_off_time)
        xr_first_sig_bar_off_time = xr.DataArray(first_sig_bar_off_time)
        
        # First bar off time
        first_bar_off_time = df_events.groupby('trial_nb').apply(event_filters.get_first_bar_off_time)
        xr_first_bar_off_time = xr.DataArray(first_bar_off_time)
        
        # Create behavioral dataset
        xr_behaviour = xr.Dataset({
            'first_reach_travel_time': xr_first_reach_travel_time,
            'last_bar_off_time': xr_last_bar_off_time,
            'first_sig_bar_off_trial_time': xr_first_sig_bar_off_time,
            'first_bar_off_trial_time': xr_first_bar_off_time
        })
        
        # Add session metadata
        session_id = df_events.attrs.get('session_id', 'unknown_session')
        xr_behaviour = xr_behaviour.expand_dims({'session_id': [session_id]})
        
        return xr_behaviour
    
    def compute_additional_behavioral_metrics(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute additional task-specific behavioral metrics.
        Can be overridden by task-specific processors.
        
        Args:
            df_events: Events dataframe
            df_conditions: Conditions dataframe
            
        Returns:
            Dict containing additional metrics
        """
        # Default implementation - empty metrics
        # Task-specific processors can override this to add custom metrics
        return {}
    
    def prepare_export_data(self, xr_photometry: xr.Dataset, df_pycontrol: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for export (e.g., to Parquet format for PlotJuggler).
        
        Args:
            xr_photometry: Photometry dataset  
            df_pycontrol: PyControl dataframe
            
        Returns:
            pd.DataFrame: Combined dataset ready for export
        """
        # Extract photometry variables with time coordinates
        var2extract = [v for v in xr_photometry.data_vars if 'time' in xr_photometry[v].coords]
        df = xr_photometry[var2extract].to_dataframe().reset_index()
        df = df.fillna(0)
        
        # Add PyControl events to the time series
        df = add_events_to_time_series(df_pycontrol, df)
        
        return df
    
    def export_to_parquet(self, df: pd.DataFrame, output_path: str, **kwargs) -> None:
        """
        Export dataframe to Parquet format.
        
        Args:
            df: DataFrame to export
            output_path: Path for output file
            **kwargs: Additional arguments for to_parquet()
        """
        default_kwargs = {'compression': 'gzip'}
        default_kwargs.update(kwargs)
        df.to_parquet(output_path, compression='gzip')
    
    def process_export_data(self, xr_photometry: xr.Dataset, df_pycontrol: pd.DataFrame, 
                           output_path: str) -> pd.DataFrame:
        """
        Complete export processing pipeline.
        
        Args:
            xr_photometry: Photometry dataset
            df_pycontrol: PyControl dataframe  
            output_path: Output file path
            
        Returns:
            pd.DataFrame: Exported dataframe (also saved to file)
        """
        # Prepare export data
        df_export = self.prepare_export_data(xr_photometry, df_pycontrol)
        
        # Export to file
        self.export_to_parquet(df_export, output_path)
        
        return df_export
    
    def process_photometry_session(self, photometry_folder: str, df_pycontrol: pd.DataFrame,
                                 df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                                 output_paths: Dict[str, str]) -> Tuple[xr.Dataset, Any]:
        """
        Complete photometry processing pipeline - the most complex processor method.
        
        This method handles the entire photometry import, processing, alignment, and 
        integration workflow that was in 04_import_pyphotometry.py.
        
        Args:
            photometry_folder: Path to folder containing .ppd files
            df_pycontrol: PyControl dataframe
            df_events: Events dataframe with trial information
            df_conditions: Conditions dataframe
            output_paths: Dict with keys 'xr_photometry', 'xr_session', 'pycontrol_aligner'
            
        Returns:
            Tuple of (final_session_dataset, pycontrol_aligner)
        """
        from pathlib import Path
        
        trial_window = df_events.attrs['trial_window']
        
        # Try to import and process photometry data
        has_photometry, dataset = self.import_and_preprocess_photometry(
            photometry_folder, df_pycontrol
        )
        
        if has_photometry:
            # Align photometry data to PyControl timeline
            dataset, pycontrol_aligner = self.align_photometry_data(
                dataset, df_events, df_pycontrol
            )
            
            # Add lick rate if lick events exist
            if any(df_events.content == 'lick'):
                xa_lick_rate = calculate_lick_rate(df_events, dataset)
                dataset['lick_rate'] = xa_lick_rate
            
            # Add event-locked data for all variables
            dataset = self.add_event_locked_data(dataset, df_events, trial_window)
            
            # Filter out pre-task data and add metadata
            dataset = dataset.sel(time=dataset.trial >= 0)
            dataset.attrs.update(df_pycontrol.attrs)
            dataset.attrs.update(df_events.attrs)
            
            # Save photometry dataset
            dataset.to_netcdf(output_paths['xr_photometry'], engine='h5netcdf')
            
        else:
            # Handle case with no photometry data
            dataset, pycontrol_aligner = self.create_no_photometry_dataset(
                df_pycontrol, df_events, trial_window, output_paths['xr_photometry']
            )
        
        # Create final session dataset with binning and condition merging
        xr_session = self.create_session_dataset(dataset, df_conditions)
        
        # Save final session dataset
        xr_session.to_netcdf(output_paths['xr_session'], engine='h5netcdf')
        
        # Save pycontrol aligner
        self.save_pycontrol_aligner(pycontrol_aligner, output_paths['pycontrol_aligner'])
        
        return xr_session, pycontrol_aligner
    
    def import_and_preprocess_photometry(self, photometry_folder: str, 
                                       df_pycontrol: pd.DataFrame) -> Tuple[bool, xr.Dataset]:
        """
        Import and preprocess photometry data from .ppd files.
        
        Args:
            photometry_folder: Path to folder containing .ppd files
            df_pycontrol: PyControl dataframe for preprocessing context
            
        Returns:
            Tuple of (has_photometry: bool, dataset: xr.Dataset or None)
        """
        from pathlib import Path
        
        try:
            pyphotometry_file = list(Path(photometry_folder).glob('*.ppd'))[0]
            
            # Import and preprocess photometry data
            data_photometry = import_ppd_auto(pyphotometry_file)
            data_photometry = preprocess_photometry(data_photometry, df_pycontrol)
            
            # Convert to xarray, skipping intermediate processing variables
            skip_var = ['analog_1_est_motion', 'time',
                       'analog_1_corrected', 'analog_1_baseline_fluo', 
                       'analog_2_baseline_fluo',
                       'isos_bleach_baseline', 'analog_1_bleach_baseline',
                       'analog_1_detrend', 'isos_detrended']
            
            dataset = photometry2xarray(data_photometry, skip_var=skip_var)
            return True, dataset
            
        except (IndexError, FileNotFoundError):
            return False, None
    
    def align_photometry_data(self, dataset: xr.Dataset, df_events: pd.DataFrame,
                            df_pycontrol: pd.DataFrame) -> Tuple[xr.Dataset, Any]:
        """
        Align photometry data to PyControl timeline using sync pulses.
        
        Args:
            dataset: Photometry dataset
            df_events: Events dataframe
            df_pycontrol: PyControl dataframe
            
        Returns:
            Tuple of (aligned_dataset, pycontrol_aligner)
        """
        # Get photometry sync pulses
        photo_rsync = dataset.attrs['pulse_times_2']
        
        # Create alignment object between photometry and PyControl
        pycontrol_aligner = create_photo_sync(df_pycontrol, dataset)
        
        # Perform the alignment
        dataset = align_photometry_to_pycontrol(dataset, df_events, pycontrol_aligner)
        
        return dataset, pycontrol_aligner
    
    def add_event_locked_data(self, dataset: xr.Dataset, df_events: pd.DataFrame,
                            trial_window: List[float]) -> xr.Dataset:
        """
        Add event-locked data for multiple variables and events.
        
        This is the complex processing loop from the original script that adds
        trial-locked data for different events and variables.
        
        Args:
            dataset: Photometry dataset
            df_events: Events dataframe
            trial_window: Trial time window [start_ms, end_ms]
            
        Returns:
            xr.Dataset: Dataset with event-locked data added
        """
        # Calculate event time coordinates
        event_period = (trial_window[1] - trial_window[0]) / 1000
        sampling_freq = dataset.attrs['sampling_rate']
        event_time_coord = np.linspace(trial_window[0], trial_window[1], 
                                     int(event_period * sampling_freq))
        
        # Determine which variables to process
        var2add = ['zscored_df_over_f']
        if 'zscored_df_over_f_analog_2' in dataset:
            var2add.append('zscored_df_over_f_analog_2')
        if 'zscored_df_over_f_analog_3' in dataset:
            var2add.append('zscored_df_over_f_analog_3')
        if 'lick_rate' in dataset:
            var2add.append('lick_rate')
        
        # Process each variable
        for var in var2add:
            # Add trigger event data
            trigger = df_events.attrs['triggers'][0]
            add_event_data(df_events, event_filters.get_first_event_from_name,
                         trial_window, dataset, event_time_coord, 
                         var, trigger, dataset.attrs['sampling_rate'],
                         filter_func_kwargs={'evt_name': trigger})
            
            # Add first bar off
            add_event_data(df_events, event_filters.get_first_bar_off, 
                         trial_window, dataset, event_time_coord, 
                         var, 'first_bar_off', dataset.attrs['sampling_rate'])

            # Add first spout
            add_event_data(df_events, event_filters.get_first_spout, 
                         trial_window, dataset, event_time_coord, 
                         var, 'first_spout', dataset.attrs['sampling_rate'])

            # Add last bar_off before first spout
            add_event_data(df_events, event_filters.get_last_bar_off_before_first_spout, 
                         trial_window, dataset, event_time_coord, 
                         var, 'last_bar_off', dataset.attrs['sampling_rate'])
            
            # Add data from any additional event triggers
            if 'extra_event_triggers' in df_events.attrs:
                for evt_triggers in df_events.attrs['extra_event_triggers']:
                    add_event_data(df_events, event_filters.get_events_from_name,
                                 trial_window, dataset, event_time_coord, 
                                 var, evt_triggers, dataset.attrs['sampling_rate'],
                                 groupby_col=None,
                                 filter_func_kwargs={'evt_name': evt_triggers})
        
        return dataset
    
    def create_no_photometry_dataset(self, df_pycontrol: pd.DataFrame, df_events: pd.DataFrame,
                                   trial_window: List[float], output_path: str) -> Tuple[xr.Dataset, None]:
        """
        Create a minimal dataset when no photometry data is available.
        
        Args:
            df_pycontrol: PyControl dataframe
            df_events: Events dataframe
            trial_window: Trial time window
            output_path: Path for dummy photometry file
            
        Returns:
            Tuple of (minimal_dataset, None)
        """
        from pathlib import Path
        
        # Create minimal dataset with PyControl data structure
        dataset = xr.Dataset()
        t = df_pycontrol.time.values
        time_coords = np.arange(t[0], t[-1])
        sampling_rate = 1000  # Default PyControl sampling rate
        
        event_period = (trial_window[1] - trial_window[0]) / 1000
        event_time_coord = np.linspace(trial_window[0], trial_window[1], 
                                     int(event_period * sampling_rate))

        dataset = dataset.assign_coords(time=('time', time_coords))
        dataset = dataset.assign_coords(event_time=('event_time', event_time_coord))
        
        # Add metadata
        dataset.attrs.update(df_pycontrol.attrs)
        dataset.attrs.update(df_events.attrs)
        dataset.attrs['sampling_rate'] = sampling_rate
        
        # Create dummy photometry file for Snakemake
        Path(output_path).touch()
        
        return dataset, None
    
    def create_session_dataset(self, dataset: xr.Dataset, df_conditions: pd.DataFrame) -> xr.Dataset:
        """
        Create final session dataset with binning and condition merging.
        
        Args:
            dataset: Photometry dataset
            df_conditions: Conditions dataframe
            
        Returns:
            xr.Dataset: Final session dataset
        """
        # Bin the data (downsample from 1000Hz to 100Hz)
        down_sample_ratio = int(dataset.attrs['sampling_rate'] / 100)
        if down_sample_ratio > 0:
            dataset_binned = dataset.coarsen(time=down_sample_ratio, 
                                           event_time=down_sample_ratio, 
                                           boundary='trim').mean()
        else:
            dataset_binned = dataset
        
        # Cast event_time to int to avoid floating point errors
        dataset_binned['event_time'] = dataset_binned.event_time.astype(int)
        dataset_binned.attrs.update(dataset.attrs)
        
        # Merge with conditions
        df_condition = df_conditions[df_conditions.index > 0]
        ds_condition = xr.Dataset.from_dataframe(df_condition)
        xr_session = xr.merge([ds_condition, dataset_binned])
        
        # Add session dimension for multi-session analysis
        session_id = dataset.attrs.get('session_id', 'unknown_session')
        xr_session = xr_session.expand_dims({'session_id': [session_id]})
        
        xr_session.attrs.update(dataset_binned.attrs)
        
        return xr_session
    
    def save_pycontrol_aligner(self, pycontrol_aligner: Any, output_path: str) -> None:
        """
        Save PyControl aligner object or create dummy file.
        
        Args:
            pycontrol_aligner: Alignment object or None
            output_path: Output file path
        """
        from pathlib import Path
        
        if pycontrol_aligner is not None:
            with open(output_path, 'wb') as f:
                pickle.dump(pycontrol_aligner, f)
        else:
            Path(output_path).touch()
    
    def process_time_warping(self, df_events_cond: pd.DataFrame, df_conditions: pd.DataFrame,
                           xr_photometry: xr.Dataset, output_path: str, figure_dir: str) -> xr.Dataset:
        """
        Complete time warping processing pipeline.
        
        This method handles the entire time warping workflow that was in 07_time_warping.py,
        including task-specific extraction specs, signal processing, and visualization.
        
        Args:
            df_events_cond: Events dataframe with conditions
            df_conditions: Conditions dataframe
            xr_photometry: Photometry dataset
            output_path: Path for output time-warped dataset
            figure_dir: Directory for saving figures
            
        Returns:
            xr.Dataset: Time-warped dataset with all signals and metadata
        """
        # Get task-specific configuration
        extraction_specs, outcome2plot = self.get_timewarp_config(df_events_cond)
        
        # Determine signals to analyze
        signal2analyze = ['zscored_df_over_f', 'zscored_df_over_f_analog_2', 'zscored_df_over_f_analog_3']
        signal2analyze = [s for s in signal2analyze if s in xr_photometry.data_vars]
        
        trigger = df_events_cond.attrs['triggers'][0]
        
        # Time warp photometry signals
        xa_list = self.time_warp_photometry_signals(
            df_events_cond, xr_photometry, signal2analyze, extraction_specs, trigger
        )
        
        # Time warp lick rate
        xa_lick_rate = self.time_warp_lick_rate(
            df_events_cond, xr_photometry, extraction_specs, trigger
        )
        xa_list.append(xa_lick_rate)
        
        # Create final dataset
        xr_conditions = xr.Dataset.from_dataframe(df_conditions)
        xr_warped = xr.merge([xr_conditions, *xa_list])
        xr_warped.attrs['extraction_specs'] = json.dumps(extraction_specs)
        
        # Save dataset
        xr_warped.to_netcdf(output_path, engine='h5netcdf')
        
        # Generate plots
        self.generate_timewarp_plots(xr_warped, signal2analyze, outcome2plot, 
                                   extraction_specs, trigger, figure_dir)
        
        # Validate results
        self.validate_timewarp_results(xr_warped)
        
        return xr_warped
    
    def get_timewarp_config(self, df_events_cond: pd.DataFrame) -> Tuple[Dict[str, Any], List[Any]]:
        """
        Get task-specific time warping configuration.
        
        Args:
            df_events_cond: Events dataframe with task metadata
            
        Returns:
            Tuple of (extraction_specs, outcome2plot)
        """
        # Load time warp specifications
        with open('params/timewarp_spec.json') as f:
            specs = json.load(f)
        
        # Get task name
        if 'task_name' in df_events_cond.attrs:
            task_name = df_events_cond.attrs['task_name']
        else:
            task_name = df_events_cond.attrs['Task name']
        
        trigger = df_events_cond.attrs['triggers'][0]
        
        # Task-specific configuration
        if task_name in ['pavlovian_spontanous_reaching_oct23',
                         'pavlovian_reaching_Oct26',
                         'pavlovian_spontanous_bar_Sep24',
                         'pavlovian_spontanous_reaching_march23',
                         'pavlovian_spontanous_reaching_oct23',
                         'pavlovian_spontanous_reaching_April24']:
            extraction_specs = specs['spontanous_reaching']
            outcome2plot = df_events_cond.trial_outcome.unique() if hasattr(df_events_cond, 'trial_outcome') else ['success']
            
        elif task_name in ['reaching_go_spout_bar_VR_Dec23',
                           'reaching_go_spout_bar_apr23',
                           'reaching_go_spout_bar_mar23',
                           'reaching_go_spout_bar_june05',
                           'reaching_go_spout_bar_nov22']:
            extraction_specs = specs['reaching_go_spout_bar_reward']
            outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
            
        elif task_name in ['reaching_go_spout_bar_VR_April24',
                           'reaching_go_spout_bar_VR_April24_silent',
                           'reaching_go_spout_bar_VR_Feb25',
                           'reaching_go_spout_bar_VR_cued_random_June25']:
            extraction_specs = specs['reaching_go_spout_bar_reward_nogap']
            outcome2plot = ['success',['omission','jackpot'],'aborted', 'no_reach', 'late_reach']
            
        elif task_name in ['reaching_go_spout_incr_break2_nov22',
                           'reaching_go_spout_incr_break2_April24',
                           'reaching_go_spout_incr_break2_Feb25',
                           'reaching_go_spout_incr_break2_June25',
                           'cued_and_cued_reward_May25']:
            extraction_specs = specs['break2']
            outcome2plot = ['success', 'no_reach', 'late_reach']
        else:
            extraction_specs = specs['default']
            # Update the trigger
            extraction_specs[trigger] = extraction_specs.pop('trigger')
            outcome2plot = df_events_cond.trial_outcome.unique() if hasattr(df_events_cond, 'trial_outcome') else ['success']
        
        return extraction_specs, outcome2plot
    
    def time_warp_photometry_signals(self, df_events_cond: pd.DataFrame, xr_photometry: xr.Dataset,
                                   signal2analyze: List[str], extraction_specs: Dict[str, Any],
                                   trigger: str) -> List[xr.DataArray]:
        """
        Time warp photometry signals using linear modeling.
        
        Args:
            df_events_cond: Events dataframe
            xr_photometry: Photometry dataset
            signal2analyze: List of signal variable names
            extraction_specs: Time warping specifications
            trigger: Trigger event name
            
        Returns:
            List of time-warped DataArrays
        """
        xa_list = []
        interp_results_list = None
        
        for signal_var in signal2analyze:
            xa, interp_results_list = lm.time_warp_data(
                df_events_cond, 
                xr_photometry[signal_var], 
                extraction_specs, 
                trigger,
                xr_photometry.attrs['sampling_rate'],
                verbose=False
            )
            xa_list.append(xa)
        
        # Add interpolation results for later plotting (from last signal)
        if interp_results_list is not None:
            df_interp_res = pd.DataFrame(interp_results_list)
            df_interp_res['trial_nb'] = xa.trial_nb
            df_interp_res = df_interp_res.set_index('trial_nb')
            xr_interp_res = df_interp_res.to_xarray()
            xa_list.append(xr_interp_res)
        
        return xa_list
    
    def time_warp_lick_rate(self, df_events_cond: pd.DataFrame, xr_photometry: xr.Dataset,
                          extraction_specs: Dict[str, Any], trigger: str) -> xr.DataArray:
        """
        Time warp lick rate signal.
        
        Args:
            df_events_cond: Events dataframe
            xr_photometry: Photometry dataset
            extraction_specs: Time warping specifications
            trigger: Trigger event name
            
        Returns:
            Time-warped lick rate DataArray
        """
        # Calculate lick rate
        lick_on = df_events_cond[df_events_cond.content == 'lick'].time
        lick_rate, _ = np.histogram(lick_on, xr_photometry.time)
        
        # Calculate rolling mean lick rate
        lick_bin_size = 0.2  # in seconds
        win_size = int(lick_bin_size * xr_photometry.attrs['sampling_rate'])
        lick_rate = np.convolve(lick_rate, np.ones(win_size)/win_size, mode='same') * xr_photometry.attrs['sampling_rate']
        
        # Create DataArray
        xa_lick_rate = xr.DataArray(
            lick_rate, 
            name='lick_rate',
            coords={'time': xr_photometry.time[:-1]},
            dims=['time']
        )
        
        # Time warp the lick rate
        xa, _ = lm.time_warp_data(
            df_events_cond, 
            xa_lick_rate, 
            extraction_specs, 
            trigger,
            xr_photometry.attrs['sampling_rate'],
            verbose=False
        )
        
        return xa
    
    def generate_timewarp_plots(self, xr_warped: xr.Dataset, signal2analyze: List[str],
                              outcome2plot: List[Any], extraction_specs: Dict[str, Any],
                              trigger: str, figure_dir: str) -> None:
        """
        Generate time warping visualization plots.
        
        Args:
            xr_warped: Time-warped dataset
            signal2analyze: List of signal variables to plot
            outcome2plot: List of outcomes to plot
            extraction_specs: Time warping specifications
            trigger: Trigger event name
            figure_dir: Directory to save figures
        """
        from pathlib import Path
        import matplotlib.pyplot as plt
        
        figure_path = Path(figure_dir)
        
        # Plot photometry signals
        for var in signal2analyze:
            fig, axes = plt.subplots(len(outcome2plot), 1, figsize=(10, 4*len(outcome2plot)))
            
            if not isinstance(axes, np.ndarray):
                axes = [axes]
            
            for outcome, ax in zip(outcome2plot, axes):
                xr2plot = xr_warped.sel(trial_nb=xr_warped.trial_outcome.isin(outcome))
                lm.plot_warpped_data(xr2plot, var, extraction_specs, trigger, ax=ax)
            
            fig.tight_layout()
            fig.savefig(figure_path / f'{var}_timewarp.png', bbox_inches='tight', dpi=200)
            plt.close(fig)
        
        # Plot lick rate
        var = 'lick_rate'
        fig, axes = plt.subplots(len(outcome2plot), 1, figsize=(10, 4*len(outcome2plot)))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for outcome, ax in zip(outcome2plot, axes):
            xr2plot = xr_warped.sel(trial_nb=xr_warped.trial_outcome.isin(outcome))
            lm.plot_warpped_data(xr2plot, var, extraction_specs, trigger, ax=ax, 
                               ylabel='Licking rate (per sec)', ylim=[0, 15])
        
        fig.tight_layout()
        fig.savefig(figure_path / f'{var}_timewarp.png', bbox_inches='tight', dpi=200)
        plt.close(fig)
    
    def validate_timewarp_results(self, xr_warped: xr.Dataset) -> None:
        """
        Validate time warping results and print statistics.
        
        Args:
            xr_warped: Time-warped dataset
        """
        # Check for valid trials
        if 'zscored_df_over_f' in xr_warped.data_vars:
            xr_success = xr_warped.sel(trial_nb=(xr_warped.trial_outcome == 'success'))
            valid_trials = np.all(~np.isnan(xr_success['zscored_df_over_f'].data), axis=1)
            valid_ratio = np.sum(valid_trials) / len(valid_trials)
            print(f'Ratio of valid successful trials: {valid_ratio:.3f}')
        else:
            print('No zscored_df_over_f signal found for validation')
    
    def compute_success(self, df_events_trials: pd.DataFrame, df_conditions: pd.DataFrame,
                       task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute success for each trial based on task-specific criteria.
        
        This is the default success logic moved from session_analysis.py.
        Task-specific processors can override this method for custom success criteria.
        
        Args:
            df_events_trials: DataFrame with trial-based event data
            df_conditions: DataFrame with trial condition information  
            task_config: Dictionary containing task configuration including:
                - task_name: Name of the behavioral task
                - triggers: List of trigger events
                - timelim: Time limits for success detection [start_ms, end_ms]
                
        Returns:
            pd.DataFrame: Conditions dataframe with 'success' column added
        """
        df_conditions = df_conditions.copy()
        df_conditions["success"] = False
        df_events = df_events_trials.copy()
        
        task_name = task_config.get('task_name', '')
        triggers = task_config.get('triggers', [])
        timelim = task_config.get('timelim', [1000, 4000])  # Default from original script

        # Go-NoGo task variants
        if task_name in [
            "reaching_go_nogo",
            "reaching_go_nogo_jc", 
            "reaching_go_nogo_opto_continuous",
            "reaching_go_nogo_opto_sinusoid",
            "reaching_go_nogo_opto_sinusoid_spout",
            "reaching_go_nogo_reversal",
            "reaching_go_nogo_reversal_incentive",
            "reaching_go_nogo_touch_spout",
        ]:
            # Go trials: success if spout event within timelim
            go_success = df_events.loc[
                (df_events[df_events.trigger == triggers[0]].index), "spout_trial_time"
            ].apply(lambda x: find_if_event_within_timelim(x, timelim))
            go_success_idx = go_success[go_success == True].index
            df_conditions.loc[(go_success_idx), "success"] = True
            
            # NoGo trials: success if no bar_off event within timelim
            nogo_success = ~df_events.loc[
                (df_events[df_events.trigger == triggers[1]].index), "bar_off_trial_time"
            ].apply(lambda x: find_if_event_within_timelim(x, timelim))
            nogo_success_idx = nogo_success[nogo_success == True].index
            df_conditions.loc[(nogo_success_idx), "success"] = True

        # Simple pavlovian Go tasks
        elif task_name in [
            "train_Go_CS-US_pavlovian",
            "reaching_yp",
            "reaching_test", 
            "reaching_test_CS",
            "train_CSgo_US_coterminated",
            "train_Go_CS-US_pavlovian",
            "train_Go_CS-US_pavlovian_with_bar",
            "pavlovian_nobar_nodelay",
        ]:
            # Success if spout event within timelim for go trials
            go_success = df_events.loc[
                (df_events[df_events.trigger == triggers[0]].index), "spout_trial_time"
            ].apply(lambda x: find_if_event_within_timelim(x, timelim))
            go_success_idx = go_success[go_success == True].index
            df_conditions.loc[(go_success_idx), "success"] = True

        # Cued-uncued go task variants
        elif task_name in ["reaching_go_spout_cued_uncued", "cued_uncued_oct22"]:
            # Reformatting trigger name for specific task
            if task_name in ["cued_uncued_oct22"]:
                df_conditions.trigger = df_conditions.trigger.str.lower()
                df_events.trigger = df_events.trigger.str.lower()

            # Cued trials: spout event within timelim
            cued_success = df_events.loc[
                (df_events[df_conditions.trigger == "cued"].index), "spout_trial_time"
            ].apply(lambda x: find_if_event_within_timelim(x, timelim))
            cued_success_idx = cued_success[cued_success == True].index

            # Uncued trials: any spout event after trial start
            uncued_success = df_events.loc[
                (df_events[df_conditions.trigger == "uncued"].index), "spout_trial_time"
            ].apply(lambda x: x[-1] > 0 if len(x) > 0 else False)
            uncued_success_idx = uncued_success[uncued_success == True].index

            df_conditions.loc[
                np.hstack((cued_success_idx.values, uncued_success_idx.values)), "success"
            ] = True

        # Simple go spout task
        elif task_name in ["reaching_go_spout_nov22"]:
            if ("spout_trial_time" in df_events.columns and 
                "US_end_timer_trial_time" in df_events.columns):
                
                reach_time_before_reward = df_events.loc[
                    :, ["spout_trial_time", "US_end_timer_trial_time"]
                ].apply(
                    lambda x: find_last_time_before_list(
                        x["spout_trial_time"], x["US_end_timer_trial_time"]
                    ),
                    axis=1,
                )
                reach_bool = reach_time_before_reward.notnull()
                reach_success_bool = reach_bool & (df_conditions.trigger == "busy_win")
                df_conditions.loc[(reach_success_bool), "success"] = True
            else:
                df_conditions["success"] = False

        # Delayed reaching tasks with bar
        elif task_name in [
            "reaching_go_spout_bar_dual_all_reward_dec22",
            "reaching_go_spout_bar_dual_dec22",
            "reaching_go_spout_bar_nov22",
            "reaching_go_spout_bar_mar23",
            "reaching_go_spout_bar_june05",
            "reaching_go_spout_bar_VR_Dec23",
            "reaching_go_spout_bar_VR_April24",
            "reaching_go_spout_bar_VR_April24_silent",
            "reaching_go_spout_bar_VR_Feb25",
            'reaching_go_spout_bar_VR_cued_random_June25'
        ]:
            if ("spout_trial_time" in df_events.columns and 
                "US_end_timer_trial_time" in df_events.columns):
                
                reach_time_before_reward = df_events.loc[
                    :, ["spout_trial_time", "US_end_timer_trial_time"]
                ].apply(
                    lambda x: find_last_time_before_list(
                        x["spout_trial_time"], x["US_end_timer_trial_time"]
                    ),
                    axis=1,
                )
                reach_bool = reach_time_before_reward.notnull()
                
                if "waiting_for_spout" in df_conditions.columns:
                    reach_success_bool = reach_bool & df_conditions.waiting_for_spout
                    df_conditions.loc[(reach_success_bool), "success"] = True
            else:
                df_conditions["success"] = False

        # Incremental break tasks
        elif task_name in [
            'cued_and_cued_reward_May25',
            'reaching_go_spout_incr_break2_Feb25',
            'reaching_go_spout_incr_break2_April24', 
            'reaching_go_spout_incr_break2_nov22',
            'reaching_go_spout_incr_break2_June25'
        ]:
            if 'US_end_timer_trial_time' in df_events.columns:
                df_conditions['success'] = ~df_events.US_end_timer_trial_time.isna()
        
        return df_conditions

    def compute_trial_outcome(self, df_conditions: pd.DataFrame, 
                             task_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute trial outcome categories for each trial.
        
        This is the default trial outcome logic moved from session_analysis.py.
        Task-specific processors can override this method for custom outcome definitions.
        
        Args:
            df_conditions: DataFrame with trial condition information
            task_config: Dictionary containing task configuration including:
                - task_name: Name of the behavioral task
                
        Returns:
            pd.DataFrame: Conditions dataframe with 'trial_outcome' column added
        """
        df_conditions = df_conditions.copy()
        task_name = task_config.get('task_name', '')
        
        # Apply trial outcome logic row by row
        df_conditions['trial_outcome'] = df_conditions.apply(
            lambda row: self._compute_single_trial_outcome(row, task_name), axis=1
        )
        
        return df_conditions
    
    def _compute_single_trial_outcome(self, row: pd.Series, task_name: str) -> str:
        """
        Compute trial outcome for a single trial row.
        
        This contains the core trial outcome logic from the original compute_trial_outcome function.
        
        Args:
            row: Single row from conditions DataFrame
            task_name: Name of the behavioral task
            
        Returns:
            str: Trial outcome category
        """
        # Basic reaching tasks with bar
        if task_name in [
            "reaching_go_spout_bar_nov22",
            "reaching_go_spout_bar_mar23", 
            "reaching_go_spout_bar_apr23",
            "reaching_go_spout_bar_june05",
            "reaching_go_spout_bar_VR_Dec23",
        ]:
            if row.break_after_abort:
                return "aborted"
            elif not row.spout:
                return "no_reach"
            elif row.button_press:
                return "button_press"
            elif row["water by bar_off"]:
                return "water_by_bar_off"
            elif row.spout and not row.water_on:
                return "late_reach"
            elif row["water by spout"]:
                return "success"
            else:
                return "undefined"

        # VR reaching tasks
        elif task_name in [
            "reaching_go_spout_bar_VR_April24",
            "reaching_go_spout_bar_VR_April24_silent",
            "reaching_go_spout_bar_VR_Feb25",
            'reaching_go_spout_bar_VR_cued_random_June25'
        ]:
            if row.break_after_abort:
                return "aborted"
            elif not row.spout:
                return "no_reach"
            elif row.button_press:
                return "button_press"
            elif row["water by bar_off"]:
                return "water_by_bar_off"
            elif row.US_jackpot:
                return "jackpot"
            elif row.US_omission:
                return "omission"
            elif row.spout and not row.water_on:
                return "late_reach"
            elif row["water by spout"]:
                return "success"
            else:
                return "undefined"

        # Incremental break tasks
        elif task_name in [
            "reaching_go_spout_incr_break2_nov22",
            "reaching_go_spout_incr_break2_Feb25",
            "reaching_go_spout_incr_break2_April24",
            "reaching_go_spout_incr_break2_Feb25",
            'reaching_go_spout_incr_break2_June25',
            'cued_and_cued_reward_May25'
        ]:
            if not row.spout:
                return "no_reach"
            elif row.button_press:
                return "button_press"
            elif row.spout and not row.US_end_timer:
                return "late_reach"
            elif row.US_end_timer:
                return "success"
            else:
                return "undefined"

        # Dual spout tasks
        elif task_name in [
            "reaching_go_spout_bar_dual_dec22",
            "reaching_go_spout_bar_dual_all_reward_dec22",
        ]:
            if row.break_after_abort:
                return "aborted"
            elif not row.spout:
                return "no_reach"
            elif row.button_press:
                return "button_press"
            elif row["water by bar_off"]:
                return "water_by_bar_off"
            elif row.spout and not row["water by spout"]:
                return "late_reach"
            elif row["water by spout"]:
                return "success"
            else:
                return "undefined"

        # Free water tasks
        elif task_name in ["reaching_go_spout_bar_free_water_june28"]:
            if row.break_after_abort:
                return "aborted"
            elif row["free reward delivered"]:
                if row.spout:
                    return "free_reward_reach"
                else:
                    return "free_reward_no_reach"
            elif not row.spout:
                return "no_reach"
            elif row.button_press:
                return "button_press"
            elif row["water by bar_off"]:
                return "water_by_bar_off"
            elif row.spout and not row.water_on:
                return "late_reach"
            elif row["water by spout"]:
                return "success"
            else:
                return "undefined"

        # Pavlovian reaching tasks
        elif task_name in [
            "pavlovian_reaching_Oct23",
            "pavlovian_reaching_Oct26",
            "pavlovian_spontanous_reaching_oct23",
            "pavlovian_spontanous_reaching_march23",
        ]:
            if row.spout:
                return "success"
            else:
                return "no_reach"

        # Default case for unknown tasks
        else:
            if row.success:
                return "success"
            else:
                return "not success"