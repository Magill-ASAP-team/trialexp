"""
Behavioral processor that combines PyControl and photometry functionality.

This processor orchestrates complete behavioral analysis workflows by combining
PyControl session processing with photometry analysis and visualization.
"""

import pandas as pd
import numpy as np
import xarray as xr
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from trialexp.process.pycontrol.utils import calculate_lick_rate
from .pycontrol_processor import PyControlProcessor
from .photometry_processor import PhotometryProcessor


class BehavioralProcessor(PyControlProcessor, PhotometryProcessor):
    """
    Combined processor for complete behavioral analysis workflows.
    
    Inherits from both PyControlProcessor and PhotometryProcessor to provide
    unified access to all processing functionality with orchestration methods.
    """
    
    def __init__(self):
        """Initialize the combined processor."""
        pass
    
    def process_full_session(self, session_path: str, tasks_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process complete PyControl session from file to analysis-ready data.
        
        Args:
            session_path: Path to PyControl session file
            tasks_df: Tasks configuration dataframe
            
        Returns:
            Tuple of (df_pycontrol, df_events, df_conditions, df_events_cond)
        """
        # Load and parse session
        df_pycontrol, task_name = self.load_and_parse_pycontrol(session_path, tasks_df)
        
        # Process lick events
        df_pycontrol = self.process_lick_events(df_pycontrol)
        
        # Get task configuration
        task_config = self.get_task_configuration(task_name, tasks_df)
        
        # Extract trial data
        df_events = self.extract_trial_data(df_pycontrol, task_config)
        
        # Compute conditions and outcomes
        df_conditions = self.compute_trial_conditions(df_events, task_config)
        df_conditions = self.compute_trial_outcome(df_events, df_conditions, task_config)
        df_conditions = self.compute_success(df_events, df_conditions, task_config)
        
        # Merge events with conditions
        df_events_cond = self.merge_events_with_conditions(df_events, df_conditions)
        
        return df_pycontrol, df_events, df_conditions, df_events_cond
    
    def process_photometry_session(self, photometry_folder: str, df_pycontrol: pd.DataFrame,
                                 df_events: pd.DataFrame, df_conditions: pd.DataFrame,
                                 output_paths: Dict[str, str]) -> Tuple[xr.Dataset, Any]:
        """
        Complete photometry processing pipeline.
        
        Args:
            photometry_folder: Path to folder containing .ppd files
            df_pycontrol: PyControl dataframe
            df_events: Events dataframe with trial information
            df_conditions: Conditions dataframe
            output_paths: Dict with keys 'xr_photometry', 'xr_session', 'pycontrol_aligner'
            
        Returns:
            Tuple of (final_session_dataset, pycontrol_aligner)
        """
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
    
    def process_time_warping(self, df_events_cond: pd.DataFrame, df_conditions: pd.DataFrame,
                           xr_photometry: xr.Dataset, output_path: str, figure_dir: str) -> xr.Dataset:
        """
        Complete time warping processing pipeline.
        
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
    
    def generate_session_plots(self, df_events_cond: pd.DataFrame, df_pycontrol: pd.DataFrame) -> Dict[str, plt.Figure]:
        """
        Generate comprehensive session plots.
        
        Args:
            df_events_cond: Events dataframe with conditions
            df_pycontrol: PyControl session dataframe
            
        Returns:
            Dict of matplotlib figures keyed by plot name
        """
        plots = {}
        
        try:
            # Event distribution histogram
            plots['event_histogram'] = self.plot_event_distribution(df_events_cond)
        except Exception as e:
            print(f"Warning: Could not generate event histogram: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Event histogram failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            plots['event_histogram'] = fig
        
        try:
            # Reach time histogram
            plots['reach_histogram'] = self.plot_reach_histogram(df_events_cond)
        except Exception as e:
            print(f"Warning: Could not generate reach histogram: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Reach histogram failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            plots['reach_histogram'] = fig
        
        try:
            # Discriminability scores
            plots['discriminability'] = self.plot_task_specific_metrics(df_events_cond, df_pycontrol)
        except Exception as e:
            print(f"Warning: Could not generate discriminability plot: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Discriminability plot failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            plots['discriminability'] = fig
        
        return plots
    
    def plot_reach_histogram(self, df_events_cond: pd.DataFrame) -> plt.Figure:
        """
        Generate reach time histogram.
        
        Args:
            df_events_cond: Events dataframe with conditions
            
        Returns:
            matplotlib Figure: Reach histogram plot
        """
        from trialexp.process.pycontrol.plot_utils import reach_time
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Calculate reach times
            reach_times = reach_time(df_events_cond)
            
            if len(reach_times) > 0:
                # Plot histogram
                ax.hist(reach_times, bins=30, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Reach Time (ms)')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Reach Times')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No reach events found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Reach Time Distribution (No Data)')
            
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Reach histogram generation failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Reach Time Distribution (Error)')
            return fig
    
    def plot_task_specific_metrics(self, df_events_cond: pd.DataFrame, df_pycontrol: pd.DataFrame) -> plt.Figure:
        """
        Generate task-specific performance metrics plot.
        
        Args:
            df_events_cond: Events dataframe with conditions
            df_pycontrol: PyControl session dataframe
            
        Returns:
            matplotlib Figure: Task metrics plot
        """
        try:
            from trialexp.process.pycontrol.utils import get_windowed_discriminability_score
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate discriminability scores over time
            window_size = 20  # trials
            xbinwidth = window_size
            binrange = (0, len(df_events_cond))
            
            scores = get_windowed_discriminability_score(
                df_events_cond, xbinwidth, binrange
            )
            
            if len(scores) > 0:
                trial_centers = np.arange(window_size//2, len(df_events_cond), window_size)
                trial_centers = trial_centers[:len(scores)]
                
                ax.plot(trial_centers, scores, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel('Trial Number')
                ax.set_ylabel('Discriminability Score')
                ax.set_title('Task Performance Over Time')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for discriminability analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Task Performance (Insufficient Data)')
            
            return fig
            
        except Exception as e:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'Task metrics plot failed:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Task Performance Metrics (Error)')
            return fig
    
    def compute_behavioral_metrics(self, df_event: pd.DataFrame) -> xr.Dataset:
        """
        Compute core behavioral metrics and return as xarray Dataset.
        
        Args:
            df_event: Event dataframe
            
        Returns:
            xr.Dataset: Behavioral metrics dataset
        """
        # Create basic behavioral metrics
        metrics = {}
        
        # Trial counts by outcome
        if 'trial_outcome' in df_event.columns:
            outcome_counts = df_event['trial_outcome'].value_counts()
            for outcome, count in outcome_counts.items():
                metrics[f'count_{outcome}'] = count
        
        # Success rate
        if 'trial_outcome' in df_event.columns:
            success_rate = (df_event['trial_outcome'] == 'success').mean()
            metrics['success_rate'] = success_rate
        
        # Total trials
        metrics['total_trials'] = len(df_event)
        
        # Convert to xarray Dataset
        xr_behaviour = xr.Dataset()
        for metric_name, value in metrics.items():
            xr_behaviour[metric_name] = xr.DataArray(value)
        
        # Add metadata
        xr_behaviour.attrs.update(df_event.attrs)
        
        return xr_behaviour
    
    def compute_additional_behavioral_metrics(self, df_event: pd.DataFrame, df_conditions: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute additional task-specific behavioral metrics.
        
        Args:
            df_event: Event dataframe
            df_conditions: Conditions dataframe
            
        Returns:
            Dict of additional metrics
        """
        additional_metrics = {}
        
        # Reaction times
        if 'spout_trial_time' in df_event.columns:
            reaction_times = []
            for idx, row in df_event.iterrows():
                spout_times = row['spout_trial_time']
                if spout_times is not None and len(spout_times) > 0:
                    reaction_times.append(spout_times[0])  # First spout time
            
            if reaction_times:
                additional_metrics['mean_reaction_time'] = np.mean(reaction_times)
                additional_metrics['std_reaction_time'] = np.std(reaction_times)
        
        # Trial outcome distribution by condition
        if 'trigger' in df_conditions.columns and 'trial_outcome' in df_conditions.columns:
            for trigger in df_conditions['trigger'].unique():
                trigger_trials = df_conditions[df_conditions['trigger'] == trigger]
                success_rate = (trigger_trials['trial_outcome'] == 'success').mean()
                additional_metrics[f'success_rate_{trigger}'] = success_rate
        
        return additional_metrics
    
    def process_export_data(self, xr_photom: xr.Dataset, df_pycontrol: pd.DataFrame, 
                          output_path: str) -> pd.DataFrame:
        """
        Process and export data to parquet format.
        
        Args:
            xr_photom: Photometry dataset
            df_pycontrol: PyControl dataframe
            output_path: Output parquet file path
            
        Returns:
            pd.DataFrame: Exported dataframe
        """
        # Prepare export data
        df_export = self.prepare_export_data(xr_photom, df_pycontrol)
        
        # Export to parquet
        self.export_to_parquet(df_export, output_path)
        
        return df_export
    
    def prepare_export_data(self, xr_photom: xr.Dataset, df_pycontrol: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for export by flattening and combining datasets.
        
        Args:
            xr_photom: Photometry dataset
            df_pycontrol: PyControl dataframe
            
        Returns:
            pd.DataFrame: Prepared export dataframe
        """
        # Convert photometry data to DataFrame
        if len(xr_photom.data_vars) > 0:
            df_photom = xr_photom.to_dataframe().reset_index()
        else:
            # Create minimal dataframe if no photometry data
            df_photom = pd.DataFrame({
                'time': df_pycontrol['time'] if 'time' in df_pycontrol.columns else range(len(df_pycontrol)),
                'trial': [0] * len(df_pycontrol)
            })
        
        # Add session metadata
        session_id = xr_photom.attrs.get('session_id', 'unknown_session')
        df_photom['session_id'] = session_id
        
        # Add task metadata
        task_name = xr_photom.attrs.get('task_name', 'unknown_task')
        df_photom['task_name'] = task_name
        
        return df_photom
    
    def export_to_parquet(self, df_export: pd.DataFrame, output_path: str) -> None:
        """
        Export dataframe to parquet format.
        
        Args:
            df_export: Dataframe to export
            output_path: Output file path
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Export to parquet with compression
        df_export.to_parquet(output_path, compression='gzip')
        
        print(f"Exported {len(df_export)} rows to {output_path}")