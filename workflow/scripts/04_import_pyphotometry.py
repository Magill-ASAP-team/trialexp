'''
This script import pyphotometry and perform the necessary processing
'''
#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.utils import *
from glob import glob
import xarray as xr
from trialexp.utils.rsync import *
import pandas as pd 
import numpy as np
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.event_filters import extract_event_time
from workflow.scripts import settings
from pathlib import Path
import pickle
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
   [settings.debug_folder + '/processed/xr_photometry.nc'],
  'import_pyphotometry')

 
#%% Load pycontrol file
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
df_event = pd.read_pickle(sinput.event_dataframe)
df_condition = pd.read_pickle(sinput.condition_dataframe)
trial_window = df_event.attrs['trial_window']

#%% Load pyphotometry file
try:
    fn = list(Path(sinput.photometry_folder).glob('*.ppd'))[0]
    has_photometry = True
    data_photometry = import_ppd(fn)
    data_photometry = denoise_filter(data_photometry, 20)
    
    # determine how to do motion correction
    animal_info = pd.read_csv('params/animal_info.csv',index_col='animal_id')
    animal_id = df_pycontrol.attrs['Subject ID'] 
    if animal_id in animal_info.index:
        injection = animal_info.loc[animal_id].injection.split(';')
        if 'RdLight' in injection:
            if not 'analog_3' in data_photometry:
                baseline_correction_multicolor(data_photometry)
                data_photometry['motion_corrected'] = 1

            else:
                # Do multicolor correction
                data_photometry = motion_correction_multicolor(data_photometry)
        else:    
            data_photometry = motion_correction_win(data_photometry)
    else:
        data_photometry = motion_correction_win(data_photometry)
    
    
    data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.01)
    data_photometry = compute_zscore(data_photometry)
    
    # Convert to xarray
    skip_var = ['analog_1_est_motion','time',
                'analog_1_corrected', 'analog_1_baseline_fluo', 
                'analog_2_baseline_fluo',
                'isos_bleach_baseline', 'analog_1_bleach_baseline',
                'analog_1_detrend', 'isos_detrended']
    
    dataset = photometry2xarray(data_photometry, skip_var = skip_var)
except IndexError:
    has_photometry = False


# %% synchornize pyphotometry with pycontrol
rsync_time = df_pycontrol[df_pycontrol.name=='rsync'].time

# Add in the relative time to different events
event_period = (trial_window[1] - trial_window[0])/1000


if has_photometry:
    sampling_freq = dataset.attrs['sampling_rate']
    event_time_coord= np.linspace(trial_window[0], trial_window[1], int(event_period*sampling_freq)) #TODO

    
    photo_rsync = dataset.attrs['pulse_times_2']

    #align pyphotometry time to pycontrol
    pycontrol_aligner = Rsync_aligner(pulse_times_A=  photo_rsync,
                    pulse_times_B= rsync_time, plot=False) #align pycontrol time to pyphotometry time


    dataset = align_photometry_to_pycontrol(dataset, df_event, pycontrol_aligner)

        
    var2add = ['zscored_df_over_f']
    if 'zscored_df_over_f_analog_2' in  dataset:
        var2add.append('zscored_df_over_f_analog_2')
        

    for var in var2add:
         # Add trigger
        trigger = df_event.attrs['triggers'][0]
        add_event_data(df_event, event_filters.get_first_event_from_name,
                    trial_window, dataset, event_time_coord, 
                    var, trigger, dataset.attrs['sampling_rate'],
                    filter_func_kwargs={'evt_name':trigger})
        
        # Add first bar off
        add_event_data(df_event, event_filters.get_first_bar_off, trial_window, dataset,event_time_coord, 
                    var, 'first_bar_off', dataset.attrs['sampling_rate'])

        # Add first spout
        add_event_data(df_event, event_filters.get_first_spout, trial_window, dataset, event_time_coord, 
                    var, 'first_spout', dataset.attrs['sampling_rate'])

        # Add last bar_off before first spout

        add_event_data(df_event, event_filters.get_last_bar_off_before_first_spout, trial_window,
                    dataset,event_time_coord, 
                    var, 'last_bar_off', dataset.attrs['sampling_rate'])


    dataset = dataset.sel(time = dataset.trial>=0) #remove data outside of task

    # add in all metadata
    dataset.attrs.update(df_pycontrol.attrs)
    dataset.attrs.update(df_event.attrs)

    dataset.to_netcdf(soutput.xr_photometry, engine='h5netcdf')

else:
    # create the dataset for pycontrol data only
    dataset = xr.Dataset()
    t = df_pycontrol.time.values
    time_coords = np.arange(t[0], t[-1])
    sampling_rate = 1000 # by default, pycontrol is samplied at 1000Hz
    event_time_coord= np.linspace(trial_window[0], trial_window[1], int(event_period*sampling_rate)) 

    dataset = dataset.assign_coords(time=('time',time_coords))
    dataset = dataset.assign_coords(event_time=('event_time', event_time_coord))
    
    # add in all metadata
    dataset.attrs.update(df_pycontrol.attrs)
    dataset.attrs.update(df_event.attrs)
    
    dataset.attrs['sampling_rate'] = sampling_rate
    
    # save a dummpy photometry file to satisfy snakemake
    Path(soutput.xr_photometry).touch()
    

# Bin the data such that we only have 1 data point per time bin
# bin according to 10ms time bin (aka 100Hz), original sampling frequency is at 1000Hz
down_sample_ratio = int(dataset.attrs['sampling_rate']/100)
dataset_binned = dataset.coarsen(time=down_sample_ratio, event_time=down_sample_ratio, boundary='trim').mean()
dataset_binned['event_time'] = dataset_binned.event_time.astype(int) #cast to int to avoid floating point error later
dataset_binned.attrs.update(dataset.attrs)

#%% Merge conditions
df_condition = df_condition[df_condition.index>0]
ds_condition = xr.Dataset.from_dataframe(df_condition)
xr_session = xr.merge([ds_condition, dataset_binned])

#add in session_id so that we can combine multiple sessions easily later
xr_session = xr_session.expand_dims({'session_id':[dataset.attrs['session_id']]})

xr_session.attrs.update(dataset_binned.attrs)

#Save the final dataset
xr_session.to_netcdf(soutput.xr_session, engine='h5netcdf')

# %%
#Also save the pyphoto_aligner
if has_photometry:
    with open(soutput.pycontrol_aligner, 'wb') as f:
        pickle.dump(pycontrol_aligner, f)
else:
    Path(soutput.pycontrol_aligner).touch()


# %%
