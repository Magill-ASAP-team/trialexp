#%%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
import xarray as xr
from matplotlib import gridspec
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.ephys.spikes_preprocessing import build_evt_fr_xarray
from trialexp.process.ephys.utils import *
from trialexp.process.group_analysis.plot_utils import style_plot
import trialexp.process.figures.plot_utils as plot_utils
from joblib import Parallel, delayed
import warnings
from workflow.scripts import settings
#%%

(sinput, soutput) = getSnake(locals(), 'workflow/spikesort.smk',
  [settings.debug_folder + r'/processed/cell_response_comparison.done'],
  'cell_response_comparison')

#%% Load inputs

xr_spikes_trials = xr.load_dataset(Path(sinput.xr_spikes_trials)) 
xr_fr = xr.load_dataset(Path(sinput.xr_spikes_fr))
session_ID = xr_spikes_trials.attrs['session_ID']
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)


#%% compute tuning prop
var2plot = [x for x in xr_spikes_trials if x.startswith('spikes_FR')]
bin_duration = xr_fr.attrs['bin_duration']
trial_window = xr_spikes_trials.attrs['trial_window']

#%%Compare with random response

style_plot()

df_list = []
for varname in var2plot:
    print('I will now do the comparison for ', varname)
    da = xr_spikes_trials[varname]
    da_rand = get_random_evt_data(xr_fr, da, trial_window)

    def compare_random(id):
        # do a mix anova test with the randomized data
        warnings.simplefilter(action='ignore', category=FutureWarning)

        data2test = create_comparison_dataframe(da_rand,da, id, varname)
        data2test = data2test[(data2test.spk_event_time>-500 ) & (data2test.spk_event_time<500)]
        comparison_result = do_mix_anova_analysis(data2test)
        
        # also plot the response curve, depending on a probability
        # because there is no need to plot all the curves
        if np.random.rand()<0.1:
            # only plot with 5% chance
            fig, ax = plt.subplots(1,1,figsize=(4,4))
            ax = sns.lineplot(data2test, x='spk_event_time', y='dv', hue='group')
            ax.set_title(f'{varname} \n {id.data}')
            ax.set_ylabel('Firing rate (Hz)')
            
            #indicate where the signifcant difference are
            yloc = ax.get_ylim()[0]
            sig_time = comparison_result['sig_interaction_time']
            ax.plot(sig_time,[yloc]*len(sig_time),'ro')
        else:
            fig = None
            
        return comparison_result, fig

    results = Parallel(n_jobs=20, verbose=10)(delayed(compare_random)(id) for id in da.cluID)
    test_results, figs = zip(*results)
    
    df_results = pd.DataFrame(test_results)
    df_results['var_name'] = varname
    df_list.append(df_results)
    
    # convert the figs to images and concentanate them
    fig2plot = [f for f in figs if f is not None]
    combined_fig = plot_utils.combine_figures(fig2plot)
    combined_fig.save(Path(soutput.response_curves_path)/f'response_curve_{varname}.png')

#%%
#%% Combine the comparison results into a dataframe and then save
df_tuning = pd.concat(df_list)
waveform_chan = get_chan_coords(xr_spikes_trials)

df_cell_prop = df_tuning.merge(waveform_chan, on='cluID')
df_cell_prop.to_pickle(Path(soutput.df_cell_prop))
