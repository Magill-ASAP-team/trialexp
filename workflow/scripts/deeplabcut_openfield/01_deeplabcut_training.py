#%% 
import deeplabcut
import os
import dotenv

#%% create training dataset
dotenv.load_dotenv()

path_config_file = os.environ['ETTIN_MOUNT_PATH']+'/Julien/ASAP/openfield/openfield-Teris-2024-01-16/config.yaml'

#%%
# deeplabcut.create_training_dataset(path_config_file)

#%%
deeplabcut.train_network(path_config_file, gputouse=1, saveiters=300, 
                         displayiters=100, maxiters=200000)
# %%
