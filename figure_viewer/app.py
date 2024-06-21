
import seaborn as sns
from trialexp.process.folder_org.utils import build_session_info_cohort, load_and_concat_dataset, filter_sessions
from shiny.express import input, render, ui, input
from shiny import reactive
from loguru import logger
from uuid import uuid4
from shiny.types import ImgData
import pandas as pd
import base64
from pathlib import Path
import settings
import os


# run figure server by
# shiny run /home/MRC.OX.AC.UK/ndcn1330/code/trialexp/figure_viewer/app.py -p 8000 -h lapis.mrc.ox.ac.uk -r

## read the cohort information

logger.debug('Reading sessions data from ettin...')
root_path = os.environ['SESSION_ROOT_DIR']
logger.debug('done')

ui.page_opts(title="ASAP figure viewer")

df_img2plot = reactive.value() # used to store the dataframe of images to plot
df_session_info = build_session_info_cohort(root_path)



@reactive.calc
def session_info():
    idx = df_session_info.cohort==input.cohort()
    if len(input.animal_id())>0:
        idx = idx & df_session_info.animal_id.isin(input.animal_id())
    if len(input.task_name())>0:
        idx = idx & df_session_info.task_name.isin(input.task_name())
    # print(df_session_info.columns)
    return df_session_info[idx].sort_values('expt_datetime')

    
with ui.sidebar(width=600):
    # df = session_info()
    df = df_session_info
    cohorts = df.cohort.unique().tolist()
    ui.input_select("cohort", "Select cohort", choices=cohorts, selected=cohorts[-1])
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
        figure_names += list((p/'processed'/'figures'/'timewarp').glob('*.png'))

    
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
        if 'timewarp' in fig_name:
            src = row.path/'processed'/'figures'/'timewarp'/fig_name
        else:
            src = row.path/'processed'/'figures'/'photometry'/fig_name
            
        if src.exists():
            img_info.append({'animal_id': row.animal_id,
                            'task_name': row.task_name,
                            'expt_datetime': row.expt_datetime,
                            'session_id': row.session_id,
                            'src': src})
            
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
                df_animal = df[df.animal_id==animal_id].sort_values('expt_datetime')
                for _, row in df_animal.iterrows():
                    p = Path(row.src)
                    if p.exists():
                        with open(p, 'rb') as f:
                            mime_type = p.suffix
                            b64_str = base64.b64encode(f.read()).decode("utf-8")
                            with ui.card():
                                ui.card_header(row.session_id)
                                ui.img(src=f"data:{mime_type};base64,{b64_str}") # create html tag directly
                        
            
