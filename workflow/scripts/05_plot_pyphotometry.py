
'''
Plotting of photometry data
'''
#%%
from snakehelper.SnakeIOHelper import getSnake
from trialexp.process.pyphotometry.plotting_utils import annotate_trial_number, plot_and_handler_error, plot_pyphoto_heatmap
from trialexp.process.pyphotometry.utils import *
from glob import glob
import xarray as xr
from trialexp.utils.rsync import *
import pandas as pd 
from scipy.interpolate import interp1d
import seaborn as sns 
from matplotlib import pyplot as plt 
import numpy as np
import os
from workflow.scripts import settings
from joblib import Parallel, delayed
import time

#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
  [settings.debug_folder + '/processed/log/photometry.done'],
  'photometry_figure')


#%%
xr_session = xr.load_dataset(sinput.xr_session)

figure_dir = soutput.trigger_photo_dir

#%% plot all event-related data

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams["legend.frameon"] = False
plt.rcParams['xtick.bottom']=True
plt.rcParams['ytick.left']=True

if os.name =='nt':
    plt.rcParams['font.family'] = ['Arial']
elif os.name =='posix':
    plt.rcParams['font.family'] = ['Lato']

sns.set_context('paper')

skip_outcome = ['button_press'] #outcome variable to skip plotting (e.g. due to having too large variance)

#%%
var2plot = [k for k in xr_session.data_vars.keys() if 'event_time' in xr_session[k].coords]

#%%

for k in var2plot:    
    print(k)
    if 'trial_nb' in xr_session[k].coords:
        df2plot = xr_session[[k,'trial_outcome']].to_dataframe().reset_index()
        df2plot = df2plot[~df2plot.trial_outcome.isin(skip_outcome)]
        
        g = sns.FacetGrid(df2plot, col='trial_outcome', col_wrap=3, hue='trial_outcome')
        g.map_dataframe(plot_and_handler_error, sns.lineplot, x='event_time', y=k)
        g.map_dataframe(annotate_trial_number)
                
        g.set_titles(col_template='{col_name}')
        g.set_xlabels('Time (ms)')
        g.figure.suptitle(xr_session.attrs['mode'], y=1.05,x=0.2)
            
        g.figure.savefig(os.path.join(figure_dir, f'{k}.png'), dpi=300, bbox_inches='tight')
    else:
        df2plot = xr_session[k].to_dataframe().reset_index()
        fig,ax = plt.subplots(1,1,figsize=(4,4), dpi=300)
        sns.lineplot(df2plot, x='event_time', y = k, ax=ax)
        ax.set_ylabel(k)
        ax.set_xlabel('Time (ms)')
        ax.set_title(xr_session.attrs['mode'])
        
        n = xr_session[k].data.shape[1]
        ax.text(0.8,0.9,f'n={n}', transform=ax.transAxes)
        
        fig.savefig(os.path.join(figure_dir, f'{k}.png'), dpi=300, bbox_inches='tight')
        
    #plot heatmap
    fig = plot_pyphoto_heatmap(xr_session[k])
    fig.savefig(os.path.join(figure_dir, f'{k}_heatmap.png'), dpi=300, bbox_inches='tight')
    
xr_session.close()