#%%
import pandas as pd 
from trialexp.process.pycontrol.plot_utils import *
from snakehelper.SnakeIOHelper import getSnake
from trialexp import config
from trialexp.process.pycontrol.processors import get_processor

#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder+'/processed/log/pycontrol.done'],'pycontrol_figures')

#%% Load processed data
df_events_cond = pd.read_pickle(sinput.event_dataframe)
df_pycontrol = pd.read_pickle(sinput.df_pycontrol)

# Get processor configuration
tasks = pd.read_csv('params/tasks_params.csv', index_col=0)
task_name = df_pycontrol.attrs['task_name']
task_idx = tasks["task"] == task_name
processor_class_name = str(tasks.loc[task_idx, 'processor_class'].iloc[0]) if 'processor_class' in tasks.columns else 'BaseTaskProcessor'
processor = get_processor(processor_class_name)

#%% Generate plots using processor
plots = processor.generate_session_plots(df_events_cond, df_pycontrol)

# %% Save all plots
plots['event_histogram'].savefig(soutput.event_histogram, dpi=300)
plots['reach_histogram'].savefig(soutput.reach_histogram, dpi=300) 
plots['discriminability'].savefig(soutput.discrim_scores, dpi=300)
