# src/seasonality_detection.py
"""
Functions to detect basic seasonality patterns and to prepare holiday/festival calendars.
"""

import pandas as pd
import numpy as np
from typing import List, Dict

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['dayofweek'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df['day_of_year'] = df['Date'].dt.dayofyear
    return df

def get_festival_calendar(df: pd.DataFrame, festival_col='Festival') -> pd.DataFrame:
    """
    Build a Prophet-style holidays DataFrame from rows where Festival != 'NA'.
    Returns DataFrame with columns ['ds','holiday'] where ds is date of festival.
    """
    if festival_col not in df.columns:
        return pd.DataFrame(columns=['ds','holiday'])

    festivals = df[df[festival_col].notnull() & (df[festival_col].astype(str)!='NA')][['Date','Festival']].drop_duplicates()
    if festivals.empty:
        return pd.DataFrame(columns=['ds','holiday'])
    cal = festivals.rename(columns={'Date':'ds','Festival':'holiday'})
    cal['ds'] = pd.to_datetime(cal['ds'])
    return cal

def detect_weekly_monthly_seasonality(df: pd.DataFrame, product: str, min_periods=180) -> Dict[str,bool]:
    """
    Quick heuristic: if we have enough history, assume weekly & monthly seasonality present.
    """
    sub = df[df['Product']==product]
    days = sub['Date'].nunique()
    return {
        'weekly': days >= 84,   # >= 12 weeks history
        'monthly': days >= 180  # >= ~6 months
    }

