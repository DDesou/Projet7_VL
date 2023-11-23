from fastapi import FastAPI,  Body
import pandas as pd
import joblib

app = FastAPI()
joblib_in = open("model.joblib","rb")
pipeline=joblib.load(joblib_in)

@app.get("/")
def read_root():
    return {"Bonjour": "Finally working!!!"}

@app.get('/prediction/')
def get_prediction(json_client: dict = Body({})):
    """
    Calculates the probability of default for a client.  
    Args:  
    - client data (json).  
    Returns:    
    - probability of default (dict).
    """
    df_one_client = pd.Series(json_client).to_frame().transpose()
    probability = pipeline[1].predict_proba(df_one_client)[:, 1][0]
    return {'probability': probability}