# src/api_app.py
"""
Minimal FastAPI app to serve forecasts for a SKU.
Designed for development/testing (Colab or local). For production deploy to a hosting service.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import os
import json

app = FastAPI(title="Inventory Forecast API")

class ForecastRequest(BaseModel):
    product: str
    horizon: int = 28

@app.get("/")
def root():
    return {"status":"ok", "message":"Inventory Forecast API"}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    # Placeholder: in dev, call demand_forecasting.train_and_forecast_per_sku
    # For now return a simple JSON structure; replace with real call in Colab
    return {"product": req.product, "horizon": req.horizon, "forecast": []}
