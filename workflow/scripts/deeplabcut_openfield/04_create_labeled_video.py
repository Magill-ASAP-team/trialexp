#%%
import deeplabcut
import os
import dotenv
import yaml
#%% create training dataset
dotenv.load_dotenv()

path_config_file = os.environ['ETTIN_MOUNT_PATH']+'/Julien/ASAP/openfield/openfield-Teris-2024-01-16/config.yaml'

with open(path_config_file,'r') as f:
    config = yaml.safe_load(f)
    
# %%

video2analyze = ['03-04-24_10-30-30.000_TT006_bottom_1.mp4',
                 '03-18-24_14-09-45.000_TT006_bottom_1.mp4',
                 '02-26-24_10-10-30.000_RE013_bottom_1.mp4',
                 '03-05-24_11-09-00.000_RE013_bottom_1.mp4']

video_paths = [config['project_path']+'/videos/'+v for v in video2analyze]

dest_folder = config['project_path']+'/videos/processed'

deeplabcut.create_labeled_video(path_config_file, 
                                video_paths[1], 
                                draw_skeleton=True,                                                               
                                destfolder=dest_folder,
                                save_frames=True,
                                outputframerate=30,
                                )

# %%
