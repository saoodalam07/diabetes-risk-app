"""
main.py  —  Diabetes Risk API (FastAPI)
---------------------------------------
Loads the trained model.pkl ONCE at startup and serves predictions
for the diabetes model (8 inputs -> Diabetic / Not Diabetic).

Run with:
    pip install fastapi uvicorn scikit-learn joblib numpy pandas
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs to test it.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import numpy as np

# The model lives in this dict so your routes can reach it after startup.
ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO 1: load the model ONCE at startup (never inside the route)
    ml_model["predictor"] = joblib.load("model.pkl")

    yield

    # TODO 2: clear the model on shutdown
    ml_model.clear()


app = FastAPI(title="Diabetes Risk API", lifespan=lifespan)


# ── Request shape: what the caller SENDS ─────────────────────────────────────
class PatientFeatures(BaseModel):
    # TODO 3: 8 fields, in the SAME order the model was trained on.
    # Field(...) bounds reject impossible values with a clean 422 error.
    Pregnancies: int = Field(..., ge=0, le=20, examples=[2])
    Glucose: float = Field(..., ge=0, le=250, examples=[120])
    BloodPressure: float = Field(..., ge=0, le=150, examples=[70])
    SkinThickness: float = Field(..., ge=0, le=100, examples=[20])
    Insulin: float = Field(..., ge=0, le=900, examples=[79])
    BMI: float = Field(..., ge=0, le=80, examples=[25.0])
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3.0, examples=[0.5])
    Age: int = Field(..., ge=1, le=120, examples=[30])


# ── Response shape: what you SEND BACK ───────────────────────────────────────
class PredictionResult(BaseModel):
    # TODO 4
    prediction: int
    result_label: str
    confidence: float


@app.get("/")
def health():
    # Done for you — a health check so you can confirm the server is up.
    return {"status": "running", "model_loaded": "predictor" in ml_model}


@app.post("/predict", response_model=PredictionResult)
def predict(features: PatientFeatures):
    model = ml_model.get("predictor")
    if model is None:
        # TODO 5
        raise HTTPException(status_code=503, detail="Model not loaded")

    # TODO 6: input array in the SAME column order as training
    X = np.array([[
        features.Pregnancies,
        features.Glucose,
        features.BloodPressure,
        features.SkinThickness,
        features.Insulin,
        features.BMI,
        features.DiabetesPedigreeFunction,
        features.Age,
    ]])

    # TODO 7: prediction (0 or 1)
    pred = int(model.predict(X)[0])

    # TODO 8: confidence = probability of the PREDICTED class
    conf = round(float(model.predict_proba(X)[0][pred]), 3)

    # TODO 9: return a PredictionResult
    return PredictionResult(
        prediction=pred,
        result_label="Diabetic" if pred == 1 else "Not Diabetic",
        confidence=conf,
    )
