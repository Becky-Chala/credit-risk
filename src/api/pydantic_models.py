from pydantic import BaseModel
from typing import List

class CustomerData(BaseModel):
    features: List[float]  # Adjust list length to match your model’s feature count

class PredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: int
