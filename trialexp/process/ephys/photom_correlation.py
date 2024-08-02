import numpy as np 
import matplotlib.pylab as plt
import pandas as pd

def plot_extrem_corr(xr_corr, xr_spike_fr_interp, xr_session, evt_name, sig_name, mode='abs'):
    '''
    Plot the average photometry signal together with the average firing rate for the largest correlations.

    Parameters:
    - xr_corr (xarray.DataArray): Xarray data array containing the correlation coefficients.
    - xr_spike_fr_interp (xarray.Dataset): Xarray dataset containing the interpolated spike firing rate data.
    - xr_session (xarray.Dataset): Xarray dataset containing the session data.
    - evt_name (str): Name of the event.
    - sig_name (str): Name of the signal.
    - mode (str, optional): How to sort the coefficient. Can be 'desc', 'asc', or 'abs'. Defaults to 'abs'.
        for 'abs', the absolute value will be sorted descendingly

    Returns:
    - None

    '''
    
    var_name = evt_name+sig_name

    # find the largest correlation
    photom_data = np.squeeze(xr_session[var_name].values)
    fr = xr_spike_fr_interp[f'spikes_FR.{evt_name}'].data
    c = xr_corr[var_name].data
    if mode == 'abs':
        idx = np.argmax(np.abs(c).data,axis=1) # the lag with the large correlation
        extrema_corr = c[np.arange(c.shape[0]),idx] #advance indexing do not work on xarray directly
        sorted_idx = np.argsort(np.abs(extrema_corr))[::-1]
    elif mode == 'asc':
        idx = np.argmin(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)
    else:
        idx = np.argmax(c.data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] 
        sorted_idx = np.argsort(extrema_corr)[::-1]
        
    lag = xr_corr.lag.data
    max_corr_loc = lag[idx]
    
    # plot average photometry signal together with average firing rate
    fig, axes = plt.subplots(3,3,figsize=(3*4,3*3))
    label1 = None
    label2 = None
    for i,ax in enumerate(axes.flat):
        ax2 = ax.twinx()
    
        if i == len(axes.flat)-1:
            label1='unit firing'
            label2='photmetry'
        
        x = fr[:,:,sorted_idx[i]].mean(axis=0)
        y = np.nanmean(photom_data,axis=0)
        shift = int(max_corr_loc[sorted_idx[i]]/(1000/50))
        ax.plot(xr_spike_fr_interp.event_time, x, label=label1)
        ax.set_ylabel('Firing rate (Hz)')
        ax2.plot(xr_spike_fr_interp.event_time, y,'r', label=label2)
        ax2.set_ylabel('dF/F')
        ax.set_title(f'Largest corr = {extrema_corr[sorted_idx[i]]:.2f} at {max_corr_loc[sorted_idx[i]]:.1f}ms')
    
    fig.tight_layout()
    fig.legend()
    
    return fig


def get_corr_spatial_distribution(xr_corr, df_metrics, signal_name):
    corr_dict={}
    for varname in xr_corr.data_vars:
        c = xr_corr[varname].data
        idx = np.argmax(np.abs(c).data,axis=1)
        extrema_corr = c[np.arange(c.shape[0]),idx] #advance indexing do not work on xarray directly
        corr_dict[varname] = extrema_corr
    
    df_corr = pd.DataFrame(corr_dict)
    df_corr['cluID'] = xr_corr.cluID;
    
    idvars = [c for c in df_corr.columns if c.endswith(signal_name)]
    
    df_corr = df_corr.merge(df_metrics,on='cluID')
    
    df_meancorr = df_corr[idvars+['ks_chan_pos_y']]
    df_meancorr = df_meancorr.groupby('ks_chan_pos_y').mean()
    
    # simplfy the column for plotting
    df_meancorr.columns = [c.replace(signal_name,'') for c in df_meancorr.columns]
    return df_meancorr
