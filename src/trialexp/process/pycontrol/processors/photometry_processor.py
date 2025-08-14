"""
Photometry processor for fiber photometry data processing and time warping functionality.

This processor handles photometry import, preprocessing, alignment with PyControl data,
event-locked data extraction, and time warping analysis.
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
from trialexp.process.pyphotometry.utils import (
    import_ppd_auto, 
    preprocess_photometry, 
    photometry2xarray, 
    add_event_data,
    align_photometry_to_pycontrol
)
from trialexp.utils.pyphotometry_utilities import create_photo_sync
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.utils import calculate_lick_rate
import trialexp.process.pyphotometry.linear_modelling as lm


class PhotometryProcessor:
    """
    Processor for photometry data processing and time warping functionality.
    
    Handles photometry import, preprocessing, alignment, event-locked analysis,
    and time warping without PyControl session dependencies.
    """
    
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
        
        This is the complex processing loop that adds trial-locked data 
        for different events and variables.
        
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
        if pycontrol_aligner is not None:
            with open(output_path, 'wb') as f:
                pickle.dump(pycontrol_aligner, f)
        else:
            Path(output_path).touch()
    
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
            fig.savefig(str(figure_path / f'{var}_timewarp.png'), bbox_inches='tight', dpi=200)
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
        fig.savefig(str(figure_path / f'{var}_timewarp.png'), bbox_inches='tight', dpi=200)
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