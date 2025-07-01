#%%
from trialexp.process.tasks.factory import get_task_analysis

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [config.debug_folder + '/processed/xr_behaviour.nc'],
  'behavorial_analysis')


# %%
df_event = pd.read_pickle(sinput.event_dataframe)
df_conditions = pd.read_pickle(sinput.condition_dataframe)

# %% Get task analysis class
task_analysis = get_task_analysis(df_event.attrs['task_name'], session_path=sinput.session_path)
xr_behaviour = task_analysis.run_behavioral_analysis(df_event, df_conditions)

# %%
xr_behaviour.to_netcdf(soutput.xr_behaviour, engine='h5netcdf')

