import seaborn as sns

# Import data from shared.py
from shared import df
from trialexp.process.folder_org.utils import build_session_info_cohort, load_and_concat_dataset, filter_sessions
from shiny.express import input, render, ui, input
from shiny import reactive
from loguru import logger
from uuid import uuid4
from shiny.types import ImgData
import pandas as pd
import base64
from pathlib import Path
## read the cohort information

logger.debug('Reading sessions data from ettin...')
root_path = '/home/MRC.OX.AC.UK/ndcn1330/ettin/Julien/ASAP/Data/'
logger.debug('done')

ui.page_opts(title="Hello sidebar!")

df_img2plot = reactive.value() # used to store the dataframe of images to plot



@reactive.calc
def session_info():
    
    session_file = Path('df_session_info.pkl')
    if session_file.exists():
        df_session_info = pd.read_pickle(session_file)
        print('Loaded existing df_session_info dataframe')
    else:
        df_session_info = build_session_info_cohort(root_path, load_pycontrol=True, pycontrol_parameters=['hold_time_required_ms'])
        df_session_info.to_pickle('df_session_info.pkl') #TODO: update the session dataframe incrementally

    idx = df_session_info.cohort==input.cohort()
    if len(input.animal_id())>0:
        idx = idx & df_session_info.animal_id.isin(input.animal_id())
    if len(input.task_name())>0:
        idx = idx & df_session_info.task_name.isin(input.task_name())
    # print(df_session_info.columns)
    return df_session_info[idx].sort_values('expt_datetime')

    
with ui.sidebar(width=500):
    df = session_info()
    cohorts = df.cohort.unique().tolist()
    ui.input_select("cohort", "Select cohort", choices=cohorts, selected=cohorts[0])
    ui.input_checkbox_group('animal_id', 'Animals', choices=[])
    ui.input_checkbox_group('task_name', 'Task', choices=[])
    ui.input_radio_buttons('figure_list', 'Figures', choices=[0,1])
    ui.input_action_button('plot_btn', 'Plot figures')

    # ui.input_switch("species", "Group by species", value=True)
    # ui.input_switch("show_rug", "Show Rug", value=True)

##################################
## Side bar controls
@reactive.effect
@reactive.event(input.cohort)
def update_animal_id():
    df = session_info()
    ui.update_checkbox_group('animal_id', choices=df.animal_id.unique().tolist())

@reactive.effect
@reactive.event(input.animal_id)
def update_task():
    df = session_info()
    ui.update_checkbox_group('task_name', choices=df.task_name.unique().tolist())
    
@reactive.effect
@reactive.event(input.task_name)
def update_figure_list():
    df = session_info()
    figure_names=[]
    for p in df.path:
        figure_names += list((p/'processed'/'figures'/'photometry').glob('*.png'))
    
    figure_names = [f.name for f in figure_names]
    figure_names = sorted(list(set(figure_names)))
    ui.update_radio_buttons('figure_list', choices=figure_names)


@reactive.effect
@reactive.event(input.plot_btn)
def update_figures_list():
    df = session_info()
    fig_name = input.figure_list()
    img_info = []
        
    # search for images in each of the session
    for _, row in df.iterrows():
        src = row.path/'processed'/'figures'/'photometry'/fig_name
        if src.exists():
            img_info.append({'animal_id': row.animal_id,
                            'task_name': row.task_name,
                            'src': row.path/'processed'/'figures'/'photometry'/fig_name})
            
    df_img2plot.set(pd.DataFrame(img_info)) # need to assign a new object so that shiny know it has been changed

    
#####################
## Figures panel
@render.express
def show_figures():
    df = df_img2plot()
    for animal_id in df.animal_id.unique():
        with ui.card():
            ui.card_header(animal_id)
            with ui.layout_column_wrap(width=1/5):
                for src in df[df.animal_id==animal_id].src:
                    p = Path(src)
                    if p.exists():
                        with open(p, 'rb') as f:
                            mime_type = p.suffix
                            b64_str = base64.b64encode(f.read()).decode("utf-8")
                            ui.img(src=f"data:{mime_type};base64,{b64_str}") # create html tag directly, TODO: change the src address
                        
            
    
# @render.plot
# def hist():
#     hue = "species" if input.species() else None
#     sns.kdeplot(df, x=input.var(), hue=hue)
#     if input.show_rug():
#         sns.rugplot(df, x=input.var(), hue=hue, color="black", alpha=0.25)

