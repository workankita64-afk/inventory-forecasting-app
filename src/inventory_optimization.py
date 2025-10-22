# src/inventory_optimization.py
"""
Compute Reorder Point (ROP), safety stock, and suggested order quantity
based on forecast and lead time/service-level inputs.
"""

import numpy as np
from scipy.stats import norm

def z_from_service_level(service_level: float) -> float:
    return float(norm.ppf(service_level))

def compute_stats_from_history(daily_series, window_days=90):
    """
    Estimate mean and std daily demand from recent history.
    daily_series: pandas Series of daily sales indexed by date or simple list
    """
    s = np.array(daily_series[-window_days:]) if len(daily_series) >= 1 else np.array([0.0])
    mean = float(np.nanmean(s))
    std = float(np.nanstd(s, ddof=0))
    return mean, std

def compute_reorder_point(mean_daily_demand: float, std_daily_demand: float, lead_time_days: int, service_level: float):
    ddl = mean_daily_demand * lead_time_days
    sigma_lead = std_daily_demand * np.sqrt(lead_time_days)
    z = z_from_service_level(service_level)
    safety_stock = z * sigma_lead
    reorder_point = ddl + safety_stock
    return {
        "mean_daily": float(mean_daily_demand),
        "std_daily": float(std_daily_demand),
        "lead_time_days": int(lead_time_days),
        "z": float(z),
        "safety_stock": float(safety_stock),
        "ddlt": float(ddl),
        "reorder_point": float(reorder_point)
    }

def suggest_order_qty(current_stock: int, reorder_point: float, forecasted_demand_horizon: float, safety_stock: float, min_order_qty:int=1):
    """
    Simple policy: when current_stock <= reorder_point, order enough to cover forecast_horizon + safety_stock - current_stock
    """
    if current_stock <= reorder_point:
        qty = max(int(np.ceil(forecasted_demand_horizon + safety_stock - current_stock)), min_order_qty)
    else:
        qty = 0
    return int(qty)

