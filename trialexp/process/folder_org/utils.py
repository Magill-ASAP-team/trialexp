'''
This contains various helper function for organizing folders and extracting meta data

'''
from glob import glob
from pathlib import Path
from datetime import datetime 
import re 
import pandas as pd 
from tqdm.auto import tqdm
import xarray as xr

def build_session_info(root_path):
    """
    This function takes a root path as input and creates a session info from the folders at that path.
    It parses the folder name with a regular expression to extract information about the animal id,
    experiment date/time, and session id. It returns a Pandas dataframe containing this information,
    as well as a calculated session number for each animal based on the experiment date/time.

    Args:
    - root_path: A string representing the root path where the session folders are located.

    Returns:
    - df_session_info: A Pandas dataframe containing the session information.
    """
    paths = glob(root_path)
    paths = [Path(p) for p in paths]

    # parse the folder name
    # TODO: extract the task name
    def parse_session_path(session_path):
        pattern = r'(\w+)-(.*)'
        session_id = session_path.name
        m = re.search(pattern, session_id)
        if m:
            animal_id = m.group(1)
            date_string = m.group(2)
            expt_datetime = datetime.strptime(date_string, "%Y-%m-%d-%H%M%S")

            return {'animal_id':animal_id, 
                    'expt_datetime': expt_datetime,
                    'session_id':session_id,
                    'path':session_path}

    session_info = [parse_session_path(p) for p in paths]
    df_session_info = pd.DataFrame(session_info)
    
    # Calculate the session number
    df_session_info['session_no'] = df_session_info.groupby('animal_id')['expt_datetime'].rank()

    return df_session_info

def load_datasets(session_paths):
    ds_list=[]
    for p in tqdm(session_paths):
        fn = p/'processed'/'xr_session.nc'
        ds = xr.open_dataset(fn) 
        ds = ds.drop_dims('time') # for performance reason
        ds_list.append(ds)
        ds.close()
        
    return ds_list


def load_and_concat_dataset(session_paths):
    ds_list = load_datasets(session_paths)
    print('Concating datasets...')
    return xr.combine_nested(ds_list,'session_id')