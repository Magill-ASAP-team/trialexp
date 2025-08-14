'''
Perform time warping to align events across trials using the processor architecture.
'''

#%%
from snakehelper.SnakeIOHelper import getSnake
import xarray as xr
import pandas as pd 
from trialexp import config
from trialexp.process.pycontrol.processors import get_processor

#%% Load inputs
(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/xr_photom_timewarped.nc'],
  'time_warping')

#%% Load processed data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)
xr_photometry = xr.load_dataset(sinput.xr_photometry, engine='h5netcdf')

# Get processor configuration
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)
task_name = df_events_cond.attrs.get('task_name', df_events_cond.attrs.get('Task name', 'unknown_task'))
task_idx = tasks["task"] == task_name
processor_class_name = str(tasks.loc[task_idx, 'processor_class'].iloc[0]) if 'processor_class' in tasks.columns and len(tasks.loc[task_idx]) > 0 else 'BaseTaskProcessor'
processor = get_processor(processor_class_name)

#%% Process time warping using processor
xr_warped = processor.process_time_warping(
    df_events_cond, df_conditions, xr_photometry, 
    soutput.xr_timewarpped, soutput.figure_dir
)
