import xarray as xr
import numpy as np 
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

def prepare_data_for_mi(fr, photom):
    # fr should be in the shsape trial x time x cluID
    # photom should be in the shape trial x time

    fr = fr.transpose([2, 0, 1]) # cluID x trial x time

    # optional smoothing
    #    atoms_smooth = savgol_filter(atoms, 21,2, axis=0)
    #     target_smooth = savgol_filter(target, 21,2, axis=0)

    # filter invalid trials
    mask_idx = (~np.isnan(fr[0, :, 0])) & (~np.isnan(photom[:,0]))
    # Filter valid trials
    fr = fr[:, mask_idx, :]
    photom = photom[mask_idx,:]

    fr_stack = fr.reshape(fr.shape[0], -1) #reshape will start from the last dimension, and keep it intact
    photom_stack = photom.ravel()
    return fr_stack, photom_stack, mask_idx

def plot_fr_with_photom(xr_session, event_win, cell_ID):
    fig, ax = plt.subplots(figsize=(10,6))
    x = xr_session['time']
    ax.plot(x, xr_session['spikes_FR_session'].sel(cluID=cell_ID).mean(dim='trial_nb'))
    ax2 = ax.twinx()
    ax2.plot(x, xr_session['zscored_df_over_f_analog_2'].mean(dim='trial_nb'), color='red')

    for e, t in event_win.items():
        ax.axvspan(t[0], t[1], alpha=0.2, edgecolor='k')
        
def extract_event_windows(extraction_specs, xr_session):
    """
    Extract time windows for each event from extraction specifications.
    If extraction is correct, trial_end should be close to the last time point
    
    Parameters
    ----------
    extraction_specs : dict
        Dictionary containing event specifications with 'event_window' and 'padding' keys
    xr_session : xr.Dataset
        Dataset containing time coordinate
        
    Returns
    -------
    dict
        Dictionary mapping event names to [start_time, end_time] windows
    """
    t = -500
    event_win = {}
    for evt, specs in extraction_specs.items():
        t -= specs['event_window'][0]
        event_win[evt] = [t + specs['event_window'][0], t + specs['event_window'][1]]
        t += specs['event_window'][1] + specs['padding']
    event_win['all'] = [xr_session.time.min().item(), xr_session.time.max().item()]
    
    return event_win

def calculate_mi_per_event(xr_session, event_win, fr_var ='spikes_FR_session', photom_var='zscored_df_over_f_analog_2'):
    """
    Calculate mutual information between neural activity and photometry signal for different event windows.
    
    Parameters
    ----------
    xr_session : xarray.Dataset
        Session data containing spikes_FR_session and photometry signals
    event_win : dict
        Dictionary mapping event names to time windows (start, end)
    photom_var : str, optional
        Name of photometry variable to analyze (default: 'zscored_df_over_f_analog_2')
    
    Returns
    -------
    xr.DataArray
        Mutual information values with dimensions [event, cluID]
    """
    t = xr_session.time
    mi_list = []
    evt_list = []
    
    for (evt, win) in event_win.items():
        if (win[1] - win[0]):  # make sure there is data
            mask = (t >= win[0]) & (t <= win[1])
            xr_region = xr_session.sel(time=mask)
            fr = xr_region[fr_var].data  # trial x time x cluID
            photom = xr_region[photom_var].data  # trial x time
            
            fr_stack, photom_stack, _ = prepare_data_for_mi(fr, photom)
            
            mi = mutual_info_regression(fr_stack.T, photom_stack, n_jobs=10)
            mi_list.append(mi)
            evt_list.append(evt)
    
    mi_list = np.stack(mi_list)
    xr_mi = xr.DataArray(mi_list, dims=['event', 'cluID'], 
                         coords={'event': evt_list, 'cluID': xr_session.cluID})
    return xr_mi

