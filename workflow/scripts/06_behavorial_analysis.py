#%%
from snakehelper.SnakeIOHelper import getSnake
import pandas as pd 
import numpy as np
from trialexp import config
from trialexp.process.pycontrol.processors import get_processor

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/xr_behaviour.nc'],
  'behavorial_analysis')


# %% Load processed data
df_event = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)

# Get processor configuration  
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)
task_name = df_event.attrs.get('task_name', 'unknown_task')
task_idx = tasks["task"] == task_name
processor_class_name = str(tasks.loc[task_idx, 'processor_class'].iloc[0]) if 'processor_class' in tasks.columns and len(tasks.loc[task_idx]) > 0 else 'BaseTaskProcessor'
processor = get_processor(processor_class_name)

#%% Compute behavioral metrics using processor
xr_behaviour = processor.compute_behavioral_metrics(df_event)

# Compute additional task-specific metrics
additional_metrics = processor.compute_additional_behavioral_metrics(df_event, df_conditions)

# Add any additional metrics to the dataset
for metric_name, metric_data in additional_metrics.items():
    if hasattr(metric_data, 'to_xarray') or isinstance(metric_data, (list, np.ndarray)):
        import xarray as xr
        xr_behaviour[metric_name] = xr.DataArray(metric_data)

# %% Save behavioral analysis results
xr_behaviour.to_netcdf(soutput.xr_behaviour, engine='h5netcdf')

