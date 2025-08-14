#%%
import pandas as pd 
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
import xarray as xr
from trialexp.process.pycontrol.processors import get_processor 
#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [config.debug_folder +'/processed/photometry.parquet'],
    'export_parquet')

# %% Load data and get processor
xr_photom = xr.load_dataset(sinput.xr_photometry)
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

# Get processor configuration
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)
task_name = df_pycontrol.attrs.get('task_name', 'unknown_task')
task_idx = tasks["task"] == task_name
processor_class_name = str(tasks.loc[task_idx, 'processor_class'].iloc[0]) if 'processor_class' in tasks.columns and len(tasks.loc[task_idx]) > 0 else 'BaseTaskProcessor'
processor = get_processor(processor_class_name)

#%% Process and export data using processor
df_export = processor.process_export_data(xr_photom, df_pycontrol, soutput.photometry_parquet)
# %%
