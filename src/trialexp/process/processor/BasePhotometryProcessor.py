from trialexp.utils.pyphotometry_utilities import create_photo_sync
from trialexp.utils.rsync import *
from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol.utils import calculate_lick_rate
from trialexp.process.pycontrol import event_filters

class BasePhotometryProcessor:
    def import_photometry_file(self, pyphotometry_file):
        data_photometry = import_ppd_auto(pyphotometry_file)
        return data_photometry
    
    def preprocess_photometry(self, data_photometry, df_pycontrol):
        data_photometry = preprocess_photometry(data_photometry, df_pycontrol)
        return data_photometry
    
    def convert2xarray(self, data_photometry):
    
        # Convert to xarray
        skip_var = ['analog_1_est_motion','time',
                    'analog_1_corrected', 'analog_1_baseline_fluo', 
                    'analog_2_baseline_fluo',
                    'isos_bleach_baseline', 'analog_1_bleach_baseline',
                    'analog_1_detrend', 'isos_detrended']
        
        dataset = photometry2xarray(data_photometry, skip_var = skip_var)
        return dataset
    
    def align_data(self, dataset, df_pycontrol, df_event):
        photo_rsync = dataset.attrs['pulse_times_2']

        #align pyphotometry time to pycontrol
        pycontrol_aligner = create_photo_sync(df_pycontrol, dataset.attrs)

        dataset = align_photometry_to_pycontrol(dataset, df_event, pycontrol_aligner)


        # Also add lick rate as another variables
        if any(df_event.content == 'lick'):
            xa_lick_rate = calculate_lick_rate(df_event, dataset)
            dataset['lick_rate'] = xa_lick_rate
            
        return dataset
    
    def process_events(self, dataset, df_event, df_pycontrol):
        trial_window = df_event.attrs['trial_window']
        event_period = (trial_window[1] - trial_window[0])/1000

        sampling_freq = dataset.attrs['sampling_rate']
        event_time_coord= np.linspace(trial_window[0], trial_window[1], int(event_period*sampling_freq)) #TODO
           
        var2add = ['zscored_df_over_f']
        if 'zscored_df_over_f_analog_2' in  dataset:
            var2add.append('zscored_df_over_f_analog_2')
            
        if 'zscored_df_over_f_analog_3' in dataset:
            var2add.append('zscored_df_over_f_analog_3')
        
        if 'lick_rate' in dataset:
            var2add.append('lick_rate')



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
        
            # Also add data from any additional event triggers
            if 'extra_event_triggers' in df_event.attrs:
                for evt_triggers in df_event.attrs['extra_event_triggers']:
                    #Note: df_event only contains event extract around the trigger
                    add_event_data(df_event, event_filters.get_events_from_name,
                        trial_window, dataset, event_time_coord, 
                        var, evt_triggers, dataset.attrs['sampling_rate'],
                        groupby_col=None,
                        filter_func_kwargs={'evt_name':evt_triggers})
                


        dataset = dataset.sel(time = dataset.trial>=0) #remove data outside of task

        # add in all metadata
        dataset.attrs.update(df_pycontrol.attrs)
        dataset.attrs.update(df_event.attrs)

        return dataset
    
    def coarsen_data(self, dataset):
        down_sample_ratio = int(dataset.attrs['sampling_rate']/100)
        if down_sample_ratio>0:
            # bin according to 10ms time bin (aka 100Hz), original sampling frequency is at 1000Hz for continuous data
            dataset_binned = dataset.coarsen(time=down_sample_ratio, event_time=down_sample_ratio, boundary='trim').mean()
        else:
            # do not resample when sampling rate is lower than 100
            dataset_binned = dataset
        dataset_binned['event_time'] = dataset_binned.event_time.astype(int) #cast to int to avoid floating point error later
        dataset_binned.attrs.update(dataset.attrs)
        
        return dataset_binned