from trialexp.process.tasks.base import TaskAnalysis
import pandas as pd
from trialexp.process.pyphotometry.utils import *
import xarray as xr
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.event_filters import extract_clean_trigger_event

class ReachingGoSpoutAnalysis(TaskAnalysis):

    def process_pycontrol(self, df_pycontrol: pd.DataFrame) -> pd.DataFrame:
        # Not implemented for this task
        return super().process_pycontrol(df_pycontrol)
    
    def run_behavioral_analysis(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame):
        # Not implemented for this task
        return super().run_behavioral_analysis(df_events, df_conditions)

    def plot_results(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame):
        # Not implemented for this task
        return super().plot_results(df_events, df_conditions)
    
    def process_photometry(self, xr_photometry: xr.Dataset, df_event: pd.DataFrame, trial_window: list) -> xr.Dataset:
        var2add = ['zscored_df_over_f']
        if 'zscored_df_over_f_analog_2' in  xr_photometry:
            var2add.append('zscored_df_over_f_analog_2')
            
        if 'zscored_df_over_f_analog_3' in xr_photometry:
            var2add.append('zscored_df_over_f_analog_3')
            
        for var in var2add:

            add_event_data(df_event, extract_clean_trigger_event, trial_window, xr_photometry, xr_photometry.event_time, 
                        var, 'clean_busy_win', xr_photometry.attrs['sampling_rate'], 
                        filter_func_kwargs = dict(clean_window = [-500,500], target_event_name='busy_win'))

            add_event_data(df_event, extract_clean_trigger_event, trial_window, xr_photometry, xr_photometry.event_time, 
                        var, 'clean_spout', xr_photometry.attrs['sampling_rate'], 
                        filter_func_kwargs = dict(clean_window = [-500,500], target_event_name='spout', ignore_events=['spout_off','bar_off']))

        return xr_photometry
