from trialexp.utils.pyphotometry_utilities import create_photo_sync
from trialexp.utils.rsync import *
from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol.utils import calculate_lick_rate

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
    
    def process_events(self):
        pass