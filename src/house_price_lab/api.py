"""Small prediction API for a previously trained House Price Lab model."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="House Price Lab", version="0.1.0")
MODEL_PATH = Path("artifacts/model.joblib")


class PropertyFeatures(BaseModel):
    area_sqft: int = Field(gt=0, le=100_000)
    bedrooms: int = Field(ge=0, le=20)
    age_years: int = Field(ge=0, le=300)
    location: str = Field(pattern="^(central|suburban|rural)$")
    property_type: str = Field(pattern="^(apartment|house|townhouse)$")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: PropertyFeatures) -> dict[str, float]:
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Model artifact is missing; run python -m house_price_lab.train first",
        )

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([features.model_dump()])
    return {"predicted_price": round(float(model.predict(row)[0]), 2)}
