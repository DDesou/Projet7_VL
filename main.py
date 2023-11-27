from fastapi import FastAPI, Body, HTTPException, Depends
from typing import Optional
import pandas as pd
import joblib
import json

app = FastAPI()

#open model
def load_model():
    model_path = "./ressources/model.joblib"
    return joblib.load(open(model_path, "rb"))

pipeline = load_model()

# open train_samp+test_samp => get all data needed
df_test = pd.read_csv('./ressources/test_samp.csv', index_col=0)
df_test = df_test.set_index('SK_ID_CURR')
ids = list(df_test.index)

df_train = pd.read_csv('./ressources/train_samp.csv', index_col=0)
df_train = df_train.set_index('SK_ID_CURR')
to_keep = ['TARGET', 'proba']
cm_met1 = df_train.drop(columns=[ele for ele in df_train if ele not in to_keep])
df_train = df_train.drop(columns=['TARGET', 'proba'])
feat = list(df_train.columns)
col = df_train.columns

df_test2 = pd.DataFrame(pipeline[0].transform(df_test[col]), columns = col)
df_test2 = df_test2.set_index(pd.Series(list(df_test.index)))

# @app decorators
@app.get("/")
def read_root():
    return {"Hello my API": "It works!!!"}

@app.get('/get_json/')
async def get_json(param_name: int):
    try:
        json_client = df_test2.loc[param_name].to_json()
        return {'data' : json_client}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Item not found with id: {id}")


@app.get('/prediction/')
def get_prediction(json_client: dict = Body({}), model: Optional[object] = Depends(load_model)):
    try:
        df_one_client = pd.Series(json_client).to_frame().transpose()
        probability = model[1].predict_proba(df_one_client)[:, 1][0]
        return {'probability': probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/get_ids")
async def get_ids():
    return {"data": ids}


@app.get("/get_feat")
async def get_feat():
    return {"data": feat}