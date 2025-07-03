from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, PredictionResponse
import mlflow.pyfunc
import pandas as pd

# Load best model from MLflow Model Registry
MODEL_NAME = "random_forest"  # <- change to your registered model name if needed
MODEL_VERSION = 1             # <- update if your model version is higher

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Credit Risk API is up and running!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    # Prepare input DataFrame
    input_df = pd.DataFrame([data.features])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of "high risk"
    
    return PredictionResponse(risk_probability=probability, is_high_risk=int(prediction))
