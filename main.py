import warnings
import numpy as np
from typing import Optional
from fastapi import FastAPI, Body, HTTPException, Depends
import pandas as pd
import joblib
#pytonimport json
import shap
shap.initjs()

#Suppress the specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
# Filter the warning related to is_sparse deprecation
warnings.filterwarnings("ignore", message="is_sparse is deprecated and will be removed in a future version.")


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

index_x0 = list(cm_met1.index[cm_met1['TARGET']==0])
index_x1 = list(cm_met1.index[cm_met1['TARGET']==1])

feat = list(df_train.columns)
col = df_train.columns

df_test2 = pd.DataFrame(pipeline[0].transform(df_test[col]), columns = col)
df_test2 = df_test2.set_index(pd.Series(list(df_test.index)))

Xtest_shap = pd.DataFrame(pipeline[0].transform(df_test[col]), columns = col, index=ids)
explainer_test = shap.TreeExplainer(pipeline[1], pipeline[0].transform(df_test), check_additivity=False)


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


@app.get('/get_test/')
async def get_test(param_name: int):
    try:
        test = df_test.loc[param_name].to_json()
        return {'data' : test}
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


@app.get("/get_kde")
async def get_kde(param_name: str):
    try:
        x0 = df_train[param_name].drop(index_x0)
        x0 = x0.dropna()
        x1 = df_train[param_name].drop(index_x1)
        x1 = x1.dropna()
        return {'feat0': x0.tolist(), 
                'feat1':x1.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get('/get_shap')
async def get_shap(param_name: int, model: Optional[object] = Depends(load_model)):
    try:
        shap_values = explainer_test(Xtest_shap.loc[[param_name]], check_additivity=False)
        # Convert SHAP values to list of ndarrays
        return {'values': shap_values.values.tolist(), 
                'feat':shap_values.feature_names, 
                'data':shap_values.data.tolist(),
                'base':shap_values.base_values.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")