import xarray as xr
import numpy as np
from sklearn import linear_model
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import Counter
from matplotlib.patches import Polygon

def extract_event(df_events, event, order, dependent_event=None):
    # extract the required event according to order, which can be one of 'first','last','last_before_first'
    # if order is 'last_before', you need to specify the depedent_event as well, it will always be
    # result is a pandas series
    events = df_events[(df_events.name==event) & (df_events.trial_time>0)]

    if len(events) == 0:
        return None

    if order =='first':
        return events.iloc[0]
    elif order == 'last':
        return events.iloc[-1]
    elif order == 'last_before_first':
        assert dependent_event is not None, 'You must supply the dependent_event'
        if (dep_event := extract_event(df_events, dependent_event, 'first')) is not None:
            df_filter = df_events[df_events.time<=dep_event.time]
            if len(events := df_filter[df_filter.name == event])>0:
                return events.iloc[-1]
        return None
    elif order == 'first_after_last':
        assert dependent_event is not None, 'You must supply the dependent_event'
        if (dep_event := extract_event(df_events, dependent_event, 'last')) is not None:
            df_filter = df_events[df_events.time>=dep_event.time]
            if len(events := df_filter[df_filter.name == event])>0:
                return events.iloc[0]
        
        return None
    else:
        raise NotImplementedError('The specified order is not supported')



def interp_data(trial_data, df_trial, trigger, extraction_specs, sampling_rate):
    # extract siganl around event by event_window in ms
    # all the time between events wall be warpped according to padding in ms
    # the trigger event must be present in df_trial

    event_specs = dict(filter(lambda k: k[0]!=trigger, extraction_specs.items()))
    trigger_specs = extraction_specs[trigger]

    # Construct the interpolation time stamp
    event_window_len = int(np.sum([v['event_window'][1]-v['event_window'][0] for k,v in extraction_specs.items()])/1000*sampling_rate)
    total_padding_len = int(np.sum([v['padding'] for k,v in extraction_specs.items()])/1000*sampling_rate)
    total_len = total_padding_len+event_window_len

    if len(trial_data) < total_len:
        raise ValueError(f'There is not enough data for interpolation required: {total_len}, available: {len(trial_data)}')
        
    t = np.zeros((total_len),)
    cur_idx = 0

    # find the trigger
    t_trigger_event = df_trial[df_trial.name == trigger]
    if len(t_trigger_event) == 0:
        raise ValueError(f"Error: the trigger {trigger} is not found")

    #copy the signal around the trigger
    t_trigger = t_trigger_event.iloc[0].time
    event_window_len = int((trigger_specs['event_window'][1] - trigger_specs['event_window'][0])/1000*sampling_rate)
    t[cur_idx:(cur_idx+event_window_len)] = np.arange(t_trigger+trigger_specs['event_window'][0], 
                                                      t_trigger+trigger_specs['event_window'][1], 1/sampling_rate*1000)
    cur_idx += event_window_len
    cur_time = t_trigger+trigger_specs['event_window'][1] #the trial time corresponding to cur_idx
    padding = trigger_specs['padding']
    padding_len = int(padding/1000*sampling_rate)
    
    interp_result = {'interp_'+trigger:True} #the trigger is always found
    

    # process the other events one by one
    # TODO  Figure out what to do if one event is missing but not the next
    for evt, specs in event_specs.items():
        dependent_event = specs.get('dependent_event', None)
        # if we can find the event, then warp from the event, if not, just start after padding
        if (event := extract_event(df_trial, evt, specs['order'], dependent_event)) is not None:
            t_event = event.time
            interp_result['interp_'+evt] = True
        else:
            t_event = cur_time+padding-specs['event_window'][0]
            interp_result['interp_'+evt] = False

            
        # find a way to warp between two events
        # Note: note there will be nan when the animal touch the spout too close to the start of next trial
        # e.g. in aborted trial
        
        # warp the inter-event period
        # Raise error if the padding is too long
        # TODO: handle this gracefully
        if cur_time + padding > t_event+specs['event_window'][0]:
            raise ValueError(f'Padding too long. {evt}  time diff: {cur_time + padding - t_event+specs["event_window"][0]}')

        t[cur_idx:(cur_idx+padding_len)] = np.linspace(cur_time, t_event+specs['event_window'][0], padding_len)
        cur_idx += padding_len
        cur_time = cur_time + padding

        # copy the data around event
        event_window_time = specs['event_window'][1] - specs['event_window'][0]
        event_window_len = int(event_window_time/1000*sampling_rate)
        t[cur_idx:(cur_idx+event_window_len)] = np.arange(t_event+specs['event_window'][0], t_event+specs['event_window'][1], 1/sampling_rate*1000)

        cur_idx += event_window_len
        cur_time = cur_time + event_window_time
        padding = specs['padding']
        padding_len = int(specs['padding']/1000*sampling_rate)
        
    # use linear interpolation to warp them
    data_interp  = trial_data.interp(time=t)
    data_interp['time'] = np.arange(total_len)/sampling_rate*1000 + trigger_specs['event_window'][0]

    return data_interp, interp_result

def extract_data(dataArray, start_time, end_time):
    # extract data specified by a start and ending time in ms
    ref_time = dataArray.time
    start_idx = np.searchsorted(ref_time, start_time)
    end_idx = np.searchsorted(ref_time, end_time)
    return dataArray[np.arange(start_idx, end_idx)]

def time_warp_data(df_events_cond, xr_signal, extraction_specs, trigger, Fs,verbose=False):
    """
    Time warps the data between events so that they can be aligned together.

    Parameters:
    - df_events_cond (pandas DataFrame): DataFrame containing event conditions.
    - xr_signal (xarray DataArray): DataArray containing the signal data.
    - extraction_specs (dict): Dictionary containing extraction specifications.
    - Fs (int): Sampling frequency.

    Returns:
    - xa (xarray DataArray): DataArray containing the time-warped data.
    """
    data_list = []
    interp_results_list = []
    
    for i in df_events_cond.trial_nb.unique():
        df_trial = df_events_cond[df_events_cond.trial_nb==i]
    
        pre_time = extraction_specs[trigger]['event_window'][0]-500
        # extract photometry data around trial
        trial_data = extract_data(xr_signal, df_trial.iloc[0].time+pre_time, df_trial.iloc[-1].time)
        
        #time wrap it
        try:
            data_p, interp_results = interp_data(trial_data, df_trial, trigger, extraction_specs, Fs)
            interp_results_list.append(interp_results)
            data_p = data_p.expand_dims({'trial_nb':[i]})
            data_list.append(data_p)
        except NotImplementedError as e:
            print(e)
        except ValueError as e:
            if verbose:
                print(f'Skipping trial {i}', e)
            
        
    xa = xr.concat(data_list,dim='trial_nb')

    return xa,interp_results_list

def add_break_mark(ax, x, y, d, w, h):
    # add break mark to the x axis
    xy = np.array([[x - w/2+d, y+h/2],
                   [x + w/2+d, y+h/2],
                   [x + w/2-d, y-h/2],
                   [x - w/2-d, y-h/2]])

    

    ax.add_patch(Polygon(xy, color='white', closed=True, zorder=3, clip_on=False))
    ax.plot([x - w/2+d, x - w/2-d], [y+h/2,  y-h/2], clip_on=False, color='k',ls='-', zorder=3)
    ax.plot([x + w/2+d, x + w/2-d], [y+h/2,  y-h/2], clip_on=False, color='k',ls='-', zorder=3)


def add_warp_info(ax, extraction_specs,trigger, adjust_ylim=True, draw_protected_region=True):
    # draw the event line and shaded region for warping info
    # add a bit of padding for text later
    ylim = ax.get_ylim()
    yrange = ylim[1] -ylim[0]
    if adjust_ylim:
        ax.set_ylim(ylim[0], ylim[1]+yrange*0.2)
    
    # plot the time point in the extraction_specs

    trigger_window = extraction_specs[trigger]['event_window']
    cur_time = trigger_window[0]
    colors = (c for c in plt.cm.tab10.colors)
    
    for i, (evt, specs) in enumerate(extraction_specs.items()):
        pre_time, post_time = specs['event_window']
        padding = specs['padding']
        
        color = next(colors)
        
        ax.axvline(cur_time-pre_time,color= color, ls='--')
        if draw_protected_region:
            ax.axvspan(cur_time, cur_time+(post_time-pre_time), alpha=0.1,color=color)
        label = specs.get('label', evt.replace('_', ' '))
        ax.text(cur_time-pre_time-10, ax.get_ylim()[1], label, rotation = 90, ha='right', va='top')
            
        cur_time += (post_time-pre_time)+padding

        ylim = ax.get_ylim()
        marker_size = (ylim[1]-ylim[0])*0.05
        if i != len(extraction_specs.keys())-1:
            add_break_mark(ax, cur_time-padding/2, ax.get_ylim()[0], 5, 20, marker_size)

    return ax

def compute_ticks(extraction_specs):
    # Calculate the tick location and labels from the specs

    ticks = []
    ticks_labels = []
    cur_time = None
    for k,v in extraction_specs.items():
        if cur_time is None:
            cur_time = v['event_window'][0]
        win_len = v['event_window'][1] - v['event_window'][0]
        
        t = [cur_time, cur_time + win_len]
        ticks += t

        
        tlab = v['event_window'][0], v['event_window'][1]
        ticks_labels += tlab
    
        cur_time =  cur_time + win_len+ v['padding']
    
    return ticks, ticks_labels

def plot_warpped_data(xa_cond, signal_var, extraction_specs,trigger, ax=None, draw_protected_region=True):
    
    palette_colors = plt.cm.tab10.colors

    df = xa_cond[[signal_var,'trial_outcome']].to_dataframe()
    
    # work with multiindex from multisession dataset
    if 'trial_id' in xa_cond.coords:
        df = df.droplevel([1,2])
        df['trial_id'] = df['session_id'].astype(str) + '_' + df['trial_nb'].astype(str)
        df = df.reset_index()    

    else:
        df = df.reset_index()
        df['trial_id'] = df['trial_nb']
    
    # if 'session_id' in df.columns:
    #         # It is a multi-session xarray Dataset, we need to clean up for later reset_index
    #         # df['trial_id'] = df.session_id.astype(str)+'_'+df.trial_nb.astype(str)
    #         df = df.drop(columns=['trial_nb','session_id'])

    df = df.dropna()
    
    if len(df)>0:
        # sometime when the event time doesn't matter the extraction_specs
        # no trial can be extracted
        
         #add in the trial number information
        df_outcome = df.groupby('trial_id').first().dropna()
        df_outcome_count = df_outcome.groupby('trial_outcome').count().time
        labels = {k:f'{k} ({df_outcome_count.loc[k]})' for k in df_outcome_count.index}
        df['trial_outcome'] = df.trial_outcome.replace(labels)
        
        outcomes = sorted(df['trial_outcome'].unique())[::-1]
        palette = {k:palette_colors[i] for i,k in enumerate(outcomes)}

        sns.lineplot(df, x='time',y=signal_var, 
                    hue='trial_outcome', palette=palette, ax = ax, n_boot=100)
        
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.25,1),title=None, frameon=False)

        
        # add a bit of padding for text later
        ylim = ax.get_ylim()
        yrange = ylim[1] -ylim[0]
        ax.set_ylim(ylim[0], ylim[1]+yrange*0.3)
        
        # plot the time point in the extraction_specs
        
        # only plot the time line if there are at least some trials that contain that event
        # idx  = df_interp_res.index.intersection(xa_cond.trial_nb)
        # event2plot = df_interp_res.loc[idx].any() #Find if there is any trial having that event
        
        trigger_window = extraction_specs[trigger]['event_window']
        cur_time = trigger_window[0]
        colors = (c for c in plt.cm.tab10.colors)
        
        for evt, specs in extraction_specs.items():
            pre_time, post_time = specs['event_window']
            padding = specs['padding']
            
            color = next(colors)
            
            if xa_cond['interp_'+evt].any():
                ax.axvline(cur_time-pre_time,color= color, ls='--')
                if draw_protected_region:
                    ax.axvspan(cur_time, cur_time+(post_time-pre_time), alpha=0.1,color=color)
                label = specs.get('label', evt.replace('_', ' '))
                ax.text(cur_time-pre_time-10, ax.get_ylim()[1], label, rotation = 90, ha='right', va='top')
                
            cur_time += (post_time-pre_time)+padding


def draw_event_line(extraction_specs,trigger, ax=None, show_label=True, draw_protected_region=False):
    
    palette_colors = plt.cm.tab10.colors


    trigger_window = extraction_specs[trigger]['event_window']
    cur_time = trigger_window[0]
    colors = (c for c in palette_colors)
    
    for evt, specs in extraction_specs.items():
        pre_time, post_time = specs['event_window']
        padding = specs['padding']
        
        color = next(colors)
        
        ax.axvline(cur_time-pre_time,color= color, ls='--')
        if draw_protected_region:
            ax.axvspan(cur_time, cur_time+(post_time-pre_time), alpha=0.1,color=color)
        
        if show_label:
            label = specs.get('label', evt.replace('_', ' '))
            ax.text(cur_time-pre_time-10, ax.get_ylim()[1], label, rotation = 90, ha='right', va='top')
        cur_time += (post_time-pre_time)+padding
        
def prepare_regression_data(xa_cond, signal_var):
    """
    Prepare the data for regression analysis.

    Parameters:
    xa_cond (xarray.DataArray): The input xarray containing the condition data.
    signal_var (str): The name of the variable to be used as the signal.

    Returns:
    tuple: A tuple containing the data array and a dictionary of predictor variables.
    """
    if 'trial_id' in xa_cond.coords:
        xr_data = xa_cond.dropna(dim='trial_id')
    else:
        xr_data = xa_cond.dropna(dim='trial_nb')

    data = np.squeeze(xr_data[signal_var].data)

    # construct the predictor index
    # trial_outcome
    event_idx = np.where(xr_data.trial_outcome == 'success')[0]
    x_event = np.zeros_like(data)
    x_event[event_idx,:] = 1

    # trial_nb (a proxy for time)
    x_trial_nb = np.tile(xr_data.trial_nb, [data.shape[1],1]).T
    
    return (data, {'trial_outcome': x_event, 'trial_nb':x_trial_nb})

def perform_linear_regression(xa_cond, data,formula, **predictor_vars):
    """
    Perform linear regression on the given data.

    Args:
        xa_cond (Xarray): Xarray object containing time data.
        data (ndarray): 2D array of shape (n_samples, n_timepoints) containing the dependent variable.
        **predictor_vars: Keyword arguments containing predictor variables as 2D arrays of shape (n_samples, n_timepoints).

    Returns:
        regress_res (DataFrame): DataFrame containing the regression results for each timepoint.
    """
    regress_res = []
    
    for t in range(data.shape[0]):
        y = data[t, :]
    
        # construct the dataframe for linear regression
        df2fit = pd.DataFrame({
            'signal': y,
        })
        
        for k, v in predictor_vars.items():
            df2fit[k] = v[t, :]
                    
        mod = smf.ols(formula=formula, data=df2fit)
        res = mod.fit()

        for factor in predictor_vars.keys():
            regress_res.append({
                'beta': res.params[factor],
                'intercept': res.params['Intercept'],  # the intercept represent the mean value
                'pvalue': res.pvalues[factor],
                'factor': factor,
                'CI': res.conf_int().loc[factor].tolist(),
                'time': xa_cond.time.data[t],
                'residual': res.resid 
            })

    regress_res = pd.DataFrame(regress_res)

    return regress_res


def highlight_pvalues(df_reg_res, ax, threshold=0.05,alpha=0.1):
    # highlight the significant time
    for _, row in df_reg_res.iterrows():
        if row.pvalue < threshold:
            ax.axvline(row.time, alpha=alpha, color='y')
            
            
def load_extraction_spec(task_name, df_conditions, specs):
    
    if task_name in ['pavlovian_spontanous_reaching_oct23',
                    'pavlovian_reaching_Oct26',
                    'pavlovian_spontanous_reaching_march23',
                    'pavlovian_spontanous_reaching_oct23']:
        
        extraction_specs = specs['spontanous_reaching']
        outcome2plot = df_conditions.trial_outcome.unique()
        
    elif task_name in ['reaching_go_spout_bar_VR_Dec23']:
        extraction_specs = specs['reaching_go_spout_bar_lick']
        outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
        
    elif task_name in ['reaching_go_spout_bar_mar23',
                    'reaching_go_spout_bar_june05',
                    'reaching_go_spout_bar_nov22',
                    'reaching_go_spout_bar_apr23']:
        extraction_specs = specs['reaching_go_spout_bar']
        outcome2plot = [['success','aborted'], 'no_reach', 'late_reach']
        
    elif task_name in ['reaching_go_spout_incr_break2_nov22']:
        extraction_specs = specs['break2']
        outcome2plot = df_conditions.trial_outcome.unique()
    else:
        extraction_specs = specs['default']
        #update the trigger
        extraction_specs[trigger] = extraction_specs.pop('trigger')
        outcome2plot = df_conditions.trial_outcome.unique()
        
    return extraction_specs, outcome2plot
