
# Script to extract waveforms and quality metrics
# for details about the metrics used, see https://spikeinterface.readthedocs.io/en/latest/modules/qualitymetrics.html
#%%
import os
from pathlib import Path

import pandas as pd
import numpy as np

import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.widgets as sw

from spikeinterface.core import select_segment_recording
from spikeinterface.postprocessing import compute_principal_components
from snakehelper.SnakeIOHelper import getSnake
import spikeinterface.extractors as se
from spikeinterface import qualitymetrics
from trialexp.process.ephys.utils import denest_string_cell, session_and_probe_specific_uid
import shutil
import settings
import time
#%% Load inputs


(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/si_quality.done'],
  'waveform_and_quality_metrics')


# %% Load Metadata and folders

sorter_name = 'kilosort'
verbose = True

rec_properties_path = Path(sinput.rec_properties)
# si_folder = Path(sinput.si_output_folder)
kilosort_folder = Path(sinput.kilosort_folder)

# Get the location of the recording
root_data_path = Path(os.environ['SORTING_ROOT_DATA_PATH'])


#%%
df_quality_metrics = []
for probe_folder in kilosort_folder.glob('Probe*'):
    
    # Check which recording we should load
    rec_properties = pd.read_csv(probe_folder.parent/'rec_prop.csv').iloc[0]
    recording_path = root_data_path/Path('/'.join(Path(rec_properties.full_path).parts[-10:-3]))
    stream = rec_properties.AP_stream
    segment_num = rec_properties.seg_index
    duration = rec_properties.duration
    probe_name = probe_folder.stem
    session_ID = probe_folder.parts[-4]
    
    
    # load sorting
    sorting = se.read_kilosort(probe_folder)
    # sorting = si.load_extractor(probe_folder.parents[1]/'si'/'kilosort3'/probe_name)
    
    # remove MUA to speed up processing later
    units2remove = sorting.unit_ids[sorting.get_property('KSLabel')=='mua']
    sorting = sorting.remove_units(units2remove)
    
    # load the correct recording
    recording = se.read_openephys(recording_path, stream_name=stream)
    recording = recording.select_segments([segment_num])
    recording.annotate(is_filtered=True)  # Neuropixel already highpass the signal at 300Hz in hardware

    # double check we have the correct recording
    assert int(recording.get_total_duration()) == int(duration), 'Error: recording durations do not match!'

    waveform_folder = probe_folder.parents[1]/'si'/probe_name
    
    if waveform_folder.exists():
        shutil.rmtree(waveform_folder)
        
    # Calculate metrics for the sorting
    analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording
                                          )
    
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=100)
    analyzer.compute("waveforms")
    analyzer.compute("templates")
    analyzer.compute("unit_locations")
    analyzer.compute("template_metrics")
    analyzer.compute("noise_levels")
    analyzer.compute(input="correlograms",
                        window_ms=50.0,
                        bin_ms=1.0,
                        method="auto")
    isi =  analyzer.compute(input="isi_histograms",
                         window_ms=50.0,
                         bin_ms=1.0,
                         method="auto")

    print('I am now calculating the quality metrics of the sorting')
    # analyzer.compute("principal_components")
    
    metric_names=['firing_rate', 'presence_ratio', 'snr', 'isi_violation', 'amplitude_cutoff']

    amp_cutoff = analyzer.compute("quality_metrics",metric_names=metric_names)
    analyzer.save_as(format='zarr', folder=waveform_folder/'analyzer')
    
    # remove waveform folder if it exists, stupid spikeinterface can't handle it itself
    # if waveform_folder.exists():
    #     shutil.rmtree(waveform_folder)
        
    # time.sleep(1)
    
    # we = si.extract_waveforms(recording, 
    #                         sorting, 
    #                         folder=waveform_folder,
    #                         max_spikes_per_unit=100, 
    #                         ms_before=1,
    #                         ms_after=2,
    #                         chunk_duration = '10s',
    #                         n_jobs=10, # too high may over-subscribe the resouces of Jade, and actually slower
    #                         progress_bar=True
    #                         )
 
    # # we = si.load_waveforms(waveform_folder) #only for debugging
    
    # # compute all other informations
    # si_correlograms, correlo_bins = si.compute_correlograms(we)
    # si_template_similarities = si.compute_template_similarity(we)
    # si_locations = si.compute_unit_locations(we)
    
    # # note: need to be using spikeinterface >0.98
    # metric_names =  qualitymetrics.get_quality_metric_list()
    # metric_names.remove('sliding_rp_violation')
    # metrics = qualitymetrics.compute_quality_metrics(we, 
    #     progress_bar = True, 
    #     n_jobs = 10,       
    #     metric_names = metric_names)


    
    # # PCA metrics are too slow, disabled for now
    # # pca = si.compute_principal_components(we, progress_bar=True, n_jobs=10)
    # # pca_metrics = ['isolation_distance','l_ratio']
    # # qualitymetrics.compute_quality_metrics(we, metric_names=pca_metrics,     
    # #                                        progress_bar = True,
    # #                                        n_jobs=10)


    # # combine all the metrics into a dataframe

    # metrics['correlograms'] = si_correlograms.tolist()
    # metrics['template_similarities'] = si_template_similarities.tolist()
    # metrics['locations'] = si_locations.tolist()
    # # metrics.attrs['correlo_bins'] = correlo_bins


    # metrics['session_ID'] = session_ID
    # metrics['probe_name'] = probe_name
    # metrics['cluster_id'] = metrics.index.values
    # metrics['cluID'] = metrics['cluster_id'].apply(lambda i: session_and_probe_specific_uid(session_ID = session_ID, probe_name = probe_name, uid = i))

    # df_quality_metrics.append(metrics)
    
#%% save output
# df_quality_metrics = pd.concat(df_quality_metrics, axis=0, ignore_index=True)
# df_quality_metrics.to_pickle(Path(soutput.df_quality_metrics))

# %%
# Combine all metrics together
units_ids = analyzer.unit_ids
metrics = {}
df2join=[]
for extension in analyzer.get_loaded_extension_names():
    wv = analyzer.get_extension(
        extension_name=extension
    )
    print(extension)
    data = wv.get_data()
    print(type(data))
    try:
        print(wv.get_data().shape)
    except:
        print(wv.get_data())
        
    if type(data) is np.ndarray and data.shape[0] == len(units_ids):
        metrics[extension] = data.tolist()
    elif type(data) is pd.core.frame.DataFrame:
        df2join.append(data)
        

df_metrics = pd.DataFrame(metrics)
df_metrics['cluster_id'] = units_ids
# %%