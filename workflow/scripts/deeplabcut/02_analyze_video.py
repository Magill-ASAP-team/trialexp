#%%
from workflow.scripts import settings
from snakehelper.SnakeIOHelper import getSnake
import deeplabcut
import numpy as np
from pathlib import Path
import shutil
import glob
import os
import trialexp.process.deeplabcut.utils as dlc_utils
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflow/deeplabcut.smk',
  [settings.debug_folder + '/processed/dlc_results.h5'],
  'analyze_video')
# %%
path_config_file = os.environ['DLC_CONFIG_PATH']
video_path = Path(os.environ['VIDEO_DIR'])

#%% read the video files
filelist = np.loadtxt(sinput.video_list,dtype=str)
side_cam = [str(video_path/(f+'.mp4')) for f in filelist if 'Side' in f]
dlc_result = Path(soutput.dlc_result)

#%%
# video = deeplabcut.DownSampleVideo(side_cam[0], width=640, outpath=str(dlc_result.parents[1]/'video'))
outpath=str(dlc_result.parents[1]/'video'/'side_downsampled.mp4')
video = dlc_utils.rescale_video(side_cam[0], output_path=outpath, width=640, frame_rate=100)

# %% analyze video
deeplabcut.analyze_videos(path_config_file, [video], gputouse=0, batchsize=16,
                          destfolder=dlc_result.parent)

# %% rename the DLC results to better work with snakemake
dlc_file = glob.glob(str(dlc_result.parent/'*.h5'))[0]
shutil.copy(dlc_file, soutput.dlc_result)
