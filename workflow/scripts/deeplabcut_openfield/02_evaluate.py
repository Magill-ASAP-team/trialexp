#%%
import deeplabcut
import os
import dotenv

#%% create training dataset
dotenv.load_dotenv()

path_config_file = os.environ['ETTIN_MOUNT_PATH']+'/Julien/ASAP/openfield/openfield-Teris-2024-01-16/config.yaml'

# %%

'''
Legends of evaluation results:
+: human labels
.: deeplabcut prediction (p> p-cutoff)
x: deepalbcut prediction (p<- p-cutoff)
'''
deeplabcut.evaluate_network(path_config_file, gputouse=1,  plotting=True)

# %%
