from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title="Iris Classifier API", description="Predict iris flower species")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris Classifier API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: IrisInput):
    try:

        features = np.array([[input_data.sepal_length,
                              input_data.sepal_width,
                              input_data.petal_length,
                              input_data.petal_width]])
        
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features).max()

        return PredictionOutput(prediction=TARGET_NAMES[prediction], confidence=round(float(proba), 3))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "LogisticRegression",
        "problem_type": "Classification",
        "features": FEATURES,
        "target_names": TARGET_NAMES
    }