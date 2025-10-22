# src/demand_forecasting.py
"""
Forecast demand using Prophet. Designed to be run per-SKU in Colab or script.
Saves forecast DataFrame and returns summary.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing import Tuple, Optional
import os
import pickle

def prepare_prophet_df(df: pd.DataFrame, target_col='Daily_Sales') -> pd.DataFrame:
    dfp = df[['Date', target_col]].rename(columns={'Date':'ds', target_col:'y'}).copy()
    dfp = dfp.sort_values('ds').dropna()
    return dfp

def fit_prophet_model(df: pd.DataFrame, holidays: Optional[pd.DataFrame] = None, extra_regressors: Optional[list]=None) -> Prophet:
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    if holidays is not None and not holidays.empty:
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, holidays=holidays)
    # add regressors
    if extra_regressors:
        for r in extra_regressors:
            m.add_regressor(r)
    m.fit(df)
    return m

def forecast_for_horizon(model: Prophet, history_df: pd.DataFrame, horizon_days: int = 28, regressors_future: dict = None) -> pd.DataFrame:
    last_date = history_df['ds'].max()
    future = pd.DataFrame({'ds': pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days)})
    if regressors_future:
        for k,v in regressors_future.items():
            # if v is scalar we repeat, if list we assign directly
            if np.isscalar(v):
                future[k] = v
            else:
                future[k] = list(v)[:horizon_days]
    forecast = model.predict(future)
    # keep required fields
    out = forecast[['ds','yhat','yhat_lower','yhat_upper']].copy()
    return out

def save_forecast(df_forecast: pd.DataFrame, out_path: str):
    df_forecast.to_csv(out_path, index=False)

def train_and_forecast_per_sku(df_all: pd.DataFrame, product: str, horizon=28, regressors: Optional[list]=None, festival_calendar: Optional[pd.DataFrame]=None):
    df = df_all[df_all['Product']==product].copy()
    if df.shape[0] < 30:
        raise ValueError(f"Not enough history for SKU {product} (rows={df.shape[0]})")
    dfp = prepare_prophet_df(df)
    # attach regressors to dfp if present
    if regressors:
        for r in regressors:
            if r in df.columns:
                dfp[r] = df[r].values
    model = fit_prophet_model(dfp, holidays=festival_calendar, extra_regressors=regressors)
    forecast = forecast_for_horizon(model, dfp, horizon_days=horizon)
    return model, forecast

if __name__ == "__main__":
    print("demand_forecasting module - import and use functions in Colab or scripts.")

