from fastapi import FastAPI, Body, HTTPException, Depends
from typing import Optional
import pandas as pd
import joblib

app = FastAPI()

def load_model():
    model_path = "./ressources/model.joblib"
    return joblib.load(open(model_path, "rb"))

pipeline = load_model()

@app.get("/")
def read_root():
    return {"Bonjour": "Now it works!!!"}

@app.get('/prediction/')
def get_prediction(json_client: dict = Body({}), model: Optional[object] = Depends(load_model)):
    """
    Calculates the probability of default for a client.  
    Args:  
    - client data (json).  
    Returns:    
    - probability of default (dict).
    """
    try:
        df_one_client = pd.Series(json_client).to_frame().transpose()
        probability = model[1].predict_proba(df_one_client)[:, 1][0]
        return {'probability': probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")