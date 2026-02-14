from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, RootModel
from typing import List, Dict
import pandas as pd
from predictor import AnomalyPredictor

app = FastAPI()
predictor = AnomalyPredictor()


# Définir le modèle de données attendu
class PredictRequest(BaseModel):
    rows: List[Dict]  # Liste de dictionnaires


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.rows:
        raise HTTPException(status_code=400, detail="No data provided")

    df = pd.DataFrame(request.rows)
    results = predictor.predict(df)

    response = {
        "model_performance": predictor.get_metrics(),
        "n_predictions": len(results),
        "results": results,
    }

    return response
