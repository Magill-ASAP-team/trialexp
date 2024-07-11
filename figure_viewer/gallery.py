from shiny import module, ui, render, reactive
from pathlib import Path
import base64
import uuid

@module.ui
def gallery_ui(animal_id):
    return ui.card(
        ui.card_header(animal_id),
        ui.output_ui('image_cards'),
    )
    
    
    
@module.server
def gallery_server(input, output, session, animal_id, df_img2plot):
        
    @output
    @render.ui
    def image_cards():
        card_list = []
        df_animal = df_img2plot()
        df_animal = df_animal[df_animal.animal_id==animal_id]
        for _, row in df_animal.iterrows():
            p = Path(row.src)
            if p.exists():
                id = str(uuid.uuid4()).replace('-','')
                card_list.append(figure_ui(id))
                figure_server(id, figure_info=row)
        
        return ui.layout_column_wrap(*card_list, width=1/4)

        
    
@module.ui
def figure_ui():
    return ui.output_ui('image_card')

@module.server
def figure_server(input, output, session, figure_info):
    @output
    @render.ui
    def image_card():
        p = Path(figure_info.src)
        with open(p, 'rb') as f:
            mime_type = p.suffix
            b64_str = base64.b64encode(f.read()).decode("utf-8")
            id = figure_info.session_id.replace('-','_')
            return  ui.card(
                            ui.card_header(ui.input_action_link('show_figure',  label=figure_info.session_id)),
                            ui.img(src=f"data:{mime_type};base64,{b64_str}") # create html tag directly
                        )
            
    
    @reactive.Effect
    @reactive.event(input.show_figure)
    def show_figure():
        p = Path(figure_info.src)
        with open(p, 'rb') as f:
            mime_type = p.suffix
            b64_str = base64.b64encode(f.read()).decode("utf-8")
            id = figure_info.session_id.replace('-','_')
            card = ui.card(
                            ui.card_header(label=figure_info.session_id),
                            ui.img(src=f"data:{mime_type};base64,{b64_str}") # create html tag directly
                        )
            
            m = ui.modal(card, size='l', easy_close=True)
            ui.modal_show(m)
    