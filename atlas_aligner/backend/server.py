import fastapi
from enum import Enum
from typing import Optional
from trialexp.process.folder_org.utils import build_session_info_cohort

root_path = '/mnt/Magill_Lab/Julien/ASAP/Data'
df_session_info = build_session_info_cohort(root_path)
# df_session_info = df_session_info.query("animal_id=='TT013'")


app = fastapi.FastAPI()
app.state.shared_obj = {'df_session_info': df_session_info}

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/{model_name}")
async def get_model(model_name:ModelName):
    return {"message": model_name}


@app.get('/sessions/')
async def get_sessions(cohort: Optional[str] = None, animal_id: Optional[str] = None):
    query_str = ''
    
    if cohort:
        query_str += f"cohort=='{cohort}'"

    if animal_id:
        query_str += f'animal_id=="{animal_id}"'
    
    df = app.state.shared_obj['df_session_info']
    
    if len(query_str) > 0:
        df = df.query(query_str)
        
    return {"message": df.session_id.to_list()}