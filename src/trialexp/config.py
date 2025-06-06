import os
from dotenv import load_dotenv
from trialexp.process.pycontrol.utils import auto_load_dotenv


# Load environment variables from .env file
load_dotenv()

# Pre-load environment variables from .env file
SNAKEMAKE_DEBUG_ROOT = os.environ.get('SNAKEMAKE_DEBUG_ROOT')
ETTIN_MOUNT_PATH = os.environ.get('ETTIN_MOUNT_PATH')
SESSION_ROOT_DIR = os.environ.get('SESSION_ROOT_DIR')
ETTIN_DATA_FOLDER = os.environ.get('ETTIN_DATA_FOLDER')
# Deeplabcut
VIDEO_DIR = os.environ.get('VIDEO_DIR')
DLC_CONFIG_PATH = os.environ.get('DLC_CONFIG_PATH')

# Sorting
SORTING_ROOT_DATA_PATH = os.environ.get('SORTING_ROOT_DATA_PATH')
CODE_ROOT_FOLDER = os.environ.get('CODE_ROOT_FOLDER')
KILOSORT3_PATH = os.environ.get('KILOSORT3_PATH')
CORTEX_LAB_SPIKES_PATH = os.environ.get('CORTEX_LAB_SPIKES_PATH')
NPY_MATLAB_PATH = os.environ.get('NPY_MATLAB_PATH')
CELL_EXPLORER_PATH = os.environ.get('CELL_EXPLORER_PATH')
RAW_DATA_ROOT_DIR = os.environ.get('RAW_DATA_ROOT_DIR')
ANATOMY_ROOT_DIR = os.environ.get('ANATOMY_ROOT_DIR')
TEMP_DATA_PATH = os.environ.get('TEMP_DATA_PATH')

