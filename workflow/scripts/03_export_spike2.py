'''
Export event data to spike2
'''
#%%
import pandas as pd 
from trialexp.process.pycontrol.utils import export_session, extract_v_line
from snakehelper.SnakeIOHelper import getSnake
from workflow.scripts import settings
from re import match
from pathlib import Path
from trialexp.process.pyphotometry.utils import *
import os
#%%

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    [settings.debug_folder +'/processed/spike2_export.done'],
    'export_spike2')

#%% Load data
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

pycontrol_time = df_pycontrol[df_pycontrol.name == 'rsync'].time

pycontrol_path = list(Path(sinput.pycontrol_folder).glob('*.txt'))[0]


v_lines, print_lines = extract_v_line(pycontrol_path)
df_dataformat = pd.read_csv('params/data_format.csv')

#%% Photometry dict

fn = list(Path(sinput.photometry_folder).glob('*.ppd'))

if fn == []:
    data_photometry = None    
else:
    fn = fn[0]
    data_format = get_dataformat(df_dataformat, df_pycontrol.attrs['session_id'])
    data_photometry = import_ppd(fn, data_format)
    data_photometry = denoise_filter(data_photometry, 20) # cannot high-pass filter the signal here
    
    
    # determine how to do motion correction
    animal_info = pd.read_csv('params/animal_info.csv',index_col='animal_id')
    animal_id = df_pycontrol.attrs['Subject ID'] #use pycontrol instead, because pyphotometry data is difficult to change manually
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

    
    data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.001)


# no down-sampling here



#%%
if fn == []:
    photometry_times_pyc = None
    photometry_keys= None
else:
    photometry_aligner = Rsync_aligner(pycontrol_time, data_photometry['pulse_times_2'])
    photometry_times_pyc = photometry_aligner.B_to_A(data_photometry['time'])
    photometry_keys =  [k for k in data_photometry.keys() if 'analog' in k]


#remove all state change event
df_pycontrol = df_pycontrol.dropna(subset='name')
df2plot = df_pycontrol[df_pycontrol.type == 'event']
# state is handled separately with export_state, whereas parameters are handled vchange_to_text

keys = df2plot.name.unique()

#%%
'''
sonpy holds a reference of the smrx file in memory, this will result in resource busy error
when it is currently opened by someone else, this will result in snakemake error which cannot be skpped. We need to handle 
the exception ourselves here
'''

spike2_path = Path(soutput.spike2_export_done).parent/'spike2.smrx'

try:
    if spike2_path.exists():
        os.remove(spike2_path)
except OSError:
    logging.warning(f'Warning: smrx file is busy. Skipping {spike2_path}')
else:
    export_session(df_pycontrol, keys, 
        data_photometry = data_photometry,
        photometry_times_pyc = photometry_times_pyc,
        photometry_keys = photometry_keys,
        print_lines = print_lines,
        v_lines = v_lines,
        smrx_filename=str(spike2_path))
    
