'''
This script imports pyphotometry and performs the necessary processing using the processor architecture.
'''
#%%
from snakehelper.SnakeIOHelper import getSnake
import pandas as pd 
from trialexp import config
from trialexp.process.pycontrol.processors import get_processor
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
   [config.debug_folder + '/processed/xr_photometry.nc'],
  'import_pyphotometry')

 
#%% Load processed data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df_events = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)

# Get processor configuration
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)
task_name = df_pycontrol.attrs.get('task_name', 'unknown_task')
task_idx = tasks["task"] == task_name
processor_class_name = str(tasks.loc[task_idx, 'processor_class'].iloc[0]) if 'processor_class' in tasks.columns and len(tasks.loc[task_idx]) > 0 else 'BaseTaskProcessor'
processor = get_processor(processor_class_name)

#%% Process photometry session using processor
output_paths = {
    'xr_photometry': soutput.xr_photometry,
    'xr_session': soutput.xr_session, 
    'pycontrol_aligner': soutput.pycontrol_aligner
}

xr_session, pycontrol_aligner = processor.process_photometry_session(
    sinput.photometry_folder, df_pycontrol, df_events, df_conditions, output_paths
)
