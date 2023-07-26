'''
This script synchronize DLC video with pycontrol and save results as xarray
'''

#%%
from trialexp.process.deeplabcut.utils import dlc2xarray, make_sync_video, marker2dataframe, plot_event_video
from workflow.scripts import settings
from snakehelper.SnakeIOHelper import getSnake
import numpy as np
from pathlib import Path
import pandas as pd
import xarray as xr
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pylab as plt
from moviepy.editor import *


(sinput, soutput) = getSnake(locals(), 'workflow/deeplabcut.smk',
  [settings.debug_folder + '/processed/xr_session_dlc.nc'],
  'sync_video')

# %% load data and convert to xarray DataArray
df_dlc = pd.read_pickle(sinput.dlc_processed)
xr_dlc = dlc2xarray(df_dlc)

# %%
xr_session = xr.open_dataset(sinput.xr_session)

xr_dlc = xr_dlc.interp(time=xr_session.time)
xr_session['dlc_markers'] = xr_dlc
xr_session.attrs['side_cam'] = df_dlc.attrs['side_cam']
xr_session.to_netcdf(soutput.xr_dlc, engine='h5netcdf')

#%% Save sync video for verification

video_path = Path(os.environ['VIDEO_DIR'])
videofile = str(video_path/df_dlc.attrs['side_cam'])
df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)
make_sync_video(videofile, soutput.synced_video, xr_session, df_pycontrol, bodypart='wrist')
