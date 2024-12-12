"""The home page of the app."""

from sidebar import styles
from sidebar.templates import template

import reflex as rx

from trialexp.process.folder_org.utils import build_session_info_cohort
from ..components.sidebar import SessionSelectState
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import xarray as xr 
from pathlib import Path
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
import numpy as np 



def min_max_downsample(data, downsample_factor):
    """
    Downsamples the input data by the given downsample factor using min-max downsampling.

    Parameters:
    data (numpy.ndarray): The input data array to be downsampled. Should be in the form (channel x time)
    downsample_factor (int): The factor by which to downsample the data. Each segment of length `downsample_factor` will be reduced to its minimum and maximum values.

    Returns:
    numpy.ndarray: A 2D array containing the downsampled data. The shape of the returned array will be (number of signals, 2 * (number of time points // downsample_factor)).
    """
    data_cropped = data[:, :data.shape[1]//downsample_factor*downsample_factor]
    min_downsampled = data_cropped.reshape(data.shape[0], -1, downsample_factor).min(axis=-1)
    max_downsampled = data_cropped.reshape(data.shape[0], -1, downsample_factor).max(axis=-1)
    downsampled_data = np.stack((min_downsampled, max_downsampled), axis=-1)
    downsampled_data = downsampled_data.reshape(data.shape[0], -1) #interleave samples
    return downsampled_data

    
root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
df_session_info = build_session_info_cohort(root_path)
df_session_info = df_session_info.query("animal_id=='TT013'")
xr_photometry = xr.open_dataset(Path(df_session_info.iloc[0].path)/'processed'/'xr_photometry.nc', engine='h5netcdf')



def plot_session(signal2plot, xr_photometry_cropped, figure_width):
            
    df_pycontrol = pd.read_pickle(Path(df_session_info.iloc[0].path)/'processed'/'df_pycontrol.pkl')
    
    # Downsample the data to fit the figure width
    ds_factor = len(xr_photometry_cropped.time)//figure_width
    print(f'ds_factor: {ds_factor}')
    data = np.stack([xr_photometry_cropped[s] for s in signal2plot])
    data_ds = min_max_downsample(data, ds_factor)
    time_ds = xr_photometry_cropped.time.data[::ds_factor].repeat(2) # min and max suppose belong to the same time

    # Doesn't work in Reflex, need to figure out how to make it work
    fig = make_subplots(rows=len(signal2plot), cols=1, shared_xaxes=True,
                        vertical_spacing=0.02)  # Reduced vertical spacing

    for idx, k in enumerate(signal2plot):
        line1 = go.Scattergl(x=time_ds/1000, 
                        y=data_ds[idx], 
                        name=f"{k} signal", 
                        mode='lines')
        fig.add_trace(line1, row=idx+1, col=1)

    fig.update_layout(
        height=200*len(signal2plot),
        width = figure_width,
        plot_bgcolor='white',  # White background
        xaxis_range = [0, time_ds[-1]/1000],
    )
    
    # Remove grid from all x and y axes
    fig.update_xaxes(showgrid=False)
    
    return fig.full_figure_for_development()


class GraphState(rx.State):
    """The state for the graph component."""

    session_id: str = ""
    figure_width:int = 1024


    fig = plot_session(['analog_1','analog_2'], xr_photometry, figure_width)
    
    @rx.event
    def refresh_graph(self):
        cohort = SessionSelectState.cohort
        session_id = SessionSelectState.session_id
        animal_id = SessionSelectState.animal_id
        
        session2plot = SessionSelectState.df_session_info.query(f"cohort=='{cohort}' & animal_id=='{animal_id}' & session_id=='{session_id}'")
        if len(session2plot)>0:
            session2plot = session2plot.iloc[0]
            

    @rx.event
    def on_relayout(self):
        #TODO: the axis range we get never got updated
        xaxis_range = self.fig.full_figure_for_development().layout.xaxis.range
        yaxis_range = self.fig.layout.yaxis.range
        print(xaxis_range)
        # Adjust the downsampling and signal range to plot based on the xaxis range
        # xr_photometry_cropped = xr_photometry.sel(time=slice(xaxis_range[0]*1000, xaxis_range[1]*1000))
        # self.fig = plot_session(['analog_1','analog_2'], xr_photometry_cropped, self.figure_width)
        
        # set the range of the figure to the previous range
        # self.fig.update_layout(xaxis_range=xaxis_range)
        # print(f'xaxis_range: {xaxis_range}')



def graph() -> rx.Component:
    return rx.plotly(data=GraphState.fig, width="80%", height="0vh", on_relayout=GraphState.on_relayout)


@template(route="/", title="Home")
def index() -> rx.Component:
    """The home page.

    Returns:
        The UI for the home page.
    """
    return graph()



    # if print_expr is not None: #TODO
    #     if isinstance(print_expr, dict):
    #         print_expr = [print_expr]

    #     for dct in print_expr:
    #         y_index += 1
    #         expr = '^\d+(?= ' + dct['expr'] + ')'
    #         list_of_match = [re.match(expr, L) for L in self.print_lines if re.match(expr, L) is not None]
    #         ts_ms = [int(m.group(0)) for m in list_of_match]
    #         line2 = go.Scatter(
    #             x=[TS_ms/1000 for TS_ms in ts_ms], y=[dct['name']] * len(ts_ms), 
    #             name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
    #         fig.add_trace(line2)

    # if event_ms is not None:
    #     if isinstance(event_ms, dict):
    #         event_ms = [event_ms]
        
    #     for dct in event_ms:
    #         y_index += 1
    #         line3 = go.Scatter(
    #             x=[t/1000 for t in dct['time_ms']],
    #             y=[dct['name']] * len(dct['time_ms']),
    #             name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
    #         fig.add_trace(line3)

    # if print_to_text:

    #     EXPR = '^(\d+)\s(.+)' #NOTE . doesn't capture \n and re.DOTALL is required below
    #     list_of_match = [re.match(EXPR, L, re.DOTALL) for L in self.print_lines if re.match(EXPR, L) is not None]
    #     ts_ms = [int(m.group(1)) for m in list_of_match]
    #     txt = [m.group(2) for m in list_of_match]

    #     # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

    #     y_index += 1
    #     txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['print_lines']*len(ts_ms), 
    #         text=txt, textposition="top center", 
    #         mode="markers", marker_symbol=symbols[y_index % 40])
    #     fig.add_trace(txtsc)

    # if vchange_to_text:
    #     EXPR = '^([1-9]\d*)\s(.+)' #NOTE Need to ignore the defaults (V 0 ****)
    #     list_of_match = [re.match(EXPR, L) for L in self.v_lines if re.match(EXPR, L) is not None]
    #     ts_ms = [int(m.group(1)) for m in list_of_match]
    #     txt = [m.group(2) for m in list_of_match]

    #     # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

    #     y_index += 1
    #     txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['V changes']*len(ts_ms), 
    #         text=txt, textposition="top center", 
    #         mode="markers", marker_symbol=symbols[y_index % 40])
    #     fig.add_trace(txtsc)



    # if state_def is not None:
    #     # Draw states as gapped lines
    #     # Assuming a list of lists of two names

    #     if isinstance(state_def, dict):# single entry
    #         state_def = [state_def]
    #         # state_ms = find_states(state_def)

    #         # line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[state_def['name']] * len(state_ms), 
    #         #     name=state_def['name'], mode='lines', line=dict(width=5))
    #         # fig.add_trace(line1)

    #     if isinstance(state_def, list):# multiple entry
    #         state_ms = None
    #         for i in state_def:
    #             assert isinstance(i, dict)
                
    #             y_index +=1
    #             state_ms = find_states(i)

    #             line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[i['name']] * len(state_ms), 
    #                 name=i['name'], mode='lines', line=dict(width=5))
    #             fig.add_trace(line1)

    #     else:
    #         state_ms = None
    # else:
    #     state_ms = None
        


    # fig.update_xaxes(title='Time (s)')
    # fig.update_yaxes(fixedrange=True) # Fix the Y axis

    # fig.update_layout(
        
    #     title =dict(
    #         text = f"{self.task_name}, {self.subject_ID} #{self.number}, on {self.datetime_string} via {self.setup_ID}"
    #     )
    # )

    # fig.show()