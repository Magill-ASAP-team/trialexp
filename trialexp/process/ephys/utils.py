
import numpy as np
import pandas as pd   
from pandas.api.types import infer_dtype
import os
import xarray as xr


def denest_string_cell(cell):
        if len(cell) == 0: 
            return 'ND'
        else:
            return str(cell[0])
        
def dataframe_cleanup(dataframe: pd.DataFrame):
    '''
    Turn object columns into str columns and fill empty gaps with ''
    '''
    types_dict = dict(zip(dataframe.columns,dataframe.dtypes))
    for (col, dtype) in types_dict.items():
        if dtype == np.dtype(object):
            dtype_inferred = infer_dtype(dataframe[col])
            dataframe[col] = dataframe[col].fillna('', downcast={np.dtype(object):str}).astype(str)
            dataframe[col] = dataframe[col].astype(dtype_inferred)
            # session_cell_metrics[col] = session_cell_metrics[col].astype(str)
    
    return dataframe

def session_and_probe_specific_uid(session_ID: str, probe_name: str, uid: int):
    '''
    Build unique cluster identifier string of cluster (UID),
    session and probe specific
    '''
    
    return session_ID + '_' + probe_name + '_' + str(uid)

def np2xrarray(x, cluID, new_dim_prefix:str):
    #Convert a numpy ndarray to xr.DataArray, taking into account the data dimension
    
    data = np.stack(x)
            
    var_new_dims = [f'{new_dim_prefix}_d{i+1}' for i in range(data.ndim-1)]
    extra_coords = {var_new_dims[i]:np.arange(data.shape[i+1]) for i in range(data.ndim-1)} # skip the first UID cooordinates
    extra_coords['cluID'] = cluID
    
    # print(name, k, data.shape, var_new_dims, extra_coords.keys())
    
    da = xr.DataArray(
        data,
        coords=extra_coords,
        dims = ['cluID',*var_new_dims]
    )
    
    
    return da

def cellmat2xarray(cell_metrics, cluID_prefix=''):
    df = pd.DataFrame()
    #convert the cell matrics struct from MATLAB to dataframe
    cell_var_names = cell_metrics.keys()
    n_row = cell_metrics['UID'].size
    
    # Reformat the cluID to be unique
    cluID = [f'{cluID_prefix}{id}' for id in cell_metrics['cluID']]
    cell_metrics.pop('UID')
    cell_metrics.pop('cluID')
    
    da_list = {}
    attrs_list = {}
    dims_dict = {}
      
    for name in cell_var_names:
        metrics = cell_metrics[name]
        if type(metrics) is np.ndarray and metrics.shape == (n_row,):
            try:
                da = np2xrarray(metrics, cluID, name)
                da_list[name] = da
            except ValueError:
                #TODO: fix the incompatibility of some object type in the attrs, preventing saving to netCDF file
                # attrs_list[name] = metrics.tolist()
                pass

        elif type(metrics) is dict:
            # More complex nested metrics, in higher dimension (e.g. 1D)
            # expand as new variable
            for k in metrics.keys():   
                if (type(metrics[k]) is np.ndarray and 
                    metrics[k].ndim==2 and 
                    metrics[k].shape[1] == n_row):
                    # 2D data
                    
                    var_new_dim = f'{k}_idx'
                    da = xr.DataArray(
                        metrics[k],
                        coords={var_new_dim:np.arange(metrics[k].shape[0]), 'cluID':cluID},
                        dims = [var_new_dim,'cluID']
                    )
                    
                    da_list[f'{name}_{k}'] = da
                    
                elif (type(metrics[k]) is np.ndarray and 
                      metrics[k].ndim==1 and 
                      metrics[k].shape[0] == n_row):
                    # more complex data, e.g. for waveforms, 3 or more dimensions
                    
                    try:
                        data = np.stack(metrics[k])
                    except ValueError:
                        # variable data format, save in attrs
                        attrs_list[f'{name}_{k}'] = metrics[k]
                        continue
                        
                    var_new_dim = f'{k}_idx'
                    
                    var_new_dims = [f'{name}_{k}_d{i+1}' for i in range(data.ndim-1)]
                    extra_coords = {var_new_dims[i]:np.arange(data.shape[i+1]) for i in range(data.ndim-1)} # skip the first UID cooordinates
                    extra_coords['cluID'] = cluID
                    
                    # print(name, k, data.shape, var_new_dims, extra_coords.keys())
                    
                    da = xr.DataArray(
                        data,
                        coords=extra_coords,
                        dims = ['cluID',*var_new_dims]
                    )
                    da_list[f'{name}_{k}'] = da
                            
    dataset = xr.Dataset(da_list)
    dataset.attrs.update(attrs_list)

    if 'general' in cell_metrics.keys():
        dataset.attrs.update(cell_metrics['general'])  
        
    if 'putativeConnections' in cell_metrics.keys():
        dataset.attrs['putativeConnections'] = cell_metrics['putativeConnections']
            
    return dataset

def cellmat2dataframe(cell_metrics):
    df = pd.DataFrame()
    #convert the cell matrics struct from MATLAB to dataframe
    cell_var_names = cell_metrics.keys()
    n_row = cell_metrics['UID'].size
    
      
    for name in cell_var_names:
        metrics = cell_metrics[name]
        if type(metrics) is np.ndarray and metrics.shape == (n_row,):
            # Save single value metrics for each cluster
            df[name] = metrics
        elif type(metrics) is dict:
            # More complex nested metrics, in higher dimension (e.g. 1D)
            # expand as new variable
            for k in metrics.keys():   
                # print(name,k)             
                if (type(metrics[k]) is np.ndarray and 
                    metrics[k].ndim==2 and 
                    metrics[k].shape[1] == n_row):
                    #1D data
                    df[f'{name}_{k}'] = metrics[k].T.tolist()
                    
                elif (type(metrics[k]) is np.ndarray and 
                      metrics[k].ndim==1 and 
                      metrics[k].shape[0] == n_row):
                    # more complex data, e.g. for waveforms
                    df[f'{name}_{k}'] = metrics[k]
                    
            
    # also save the generate properties
    if 'general' in cell_metrics.keys():
        df.attrs.update(cell_metrics['general'])  
        
    if 'putativeConnections' in cell_metrics.keys():
        df.attrs['putativeConnections'] = cell_metrics['putativeConnections']
            
    return df



def prepare_mathlab_path(eng):
    # Adding Path to Matlab from Environment variables defined in .env file.
    s = eng.genpath(os.environ['CORTEX_LAB_SPIKES_PATH']) # maybe unnecessary, just open all ks3 results
    n = eng.genpath(os.environ['NPY_MATLAB_PATH'])
    c = eng.genpath(os.environ['CELL_EXPLORER_PATH'])

    eng.addpath(s, nargout=0)
    eng.addpath(n, nargout=0)
    eng.addpath(c, nargout=0)