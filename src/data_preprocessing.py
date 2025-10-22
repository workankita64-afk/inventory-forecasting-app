# src/data_preprocessing.py
"""
Utilities to load dataset from Google Drive (or local) and do initial cleaning.
Designed to be run in Google Colab (mount drive) or any Python env.
"""

import pandas as pd
from typing import Tuple

def load_csv_from_path(path: str, parse_dates=['Date']) -> pd.DataFrame:
    """
    Load CSV into dataframe. 'path' can be a Google Drive path when drive is mounted,
    e.g. '/content/drive/MyDrive/inventory_sales.csv'
    """
    df = pd.read_csv(path, parse_dates=parse_dates)
    return df

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure column types
    df = df.copy()
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # Ensure Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    # Numeric conversions and fillna defaults
    numeric_cols = [
        'Daily_Sales','Opening_Stock','Closing_Stock','Replenishment_Qty',
        'New_Order_Qty','Stock_Coverage_Days','Service_Level','Price_Index',
        'Threshold','Lead_Time_Days','Regional_Weight','Weather_Factor',
        'Weekend_Factor','Promotion_Factor','Economic_Factor','Adjusted_Sales',
        'Inventory_Turnover','Inventory_Turnover_Rate','Efficiency_Score','Profit_Margin'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Fill string columns
    for c in ['Product','Region','Festival','Stock_Out_Flag','Stock_Recommendation']:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna('NA')

    return df

def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    If dataset has transaction-level data, aggregate to daily per SKU.
    If already daily, this will return similar data grouped by Product+Date.
    """
    key_cols = ['Date','Product','Region']
    agg_cols = {
        'Daily_Sales':'sum',
        'Opening_Stock':'max',
        'Closing_Stock':'max',
        'Replenishment_Qty':'sum',
        'New_Order_Qty':'sum',
        'Stock_Out_Flag': lambda x: x.astype(str).mode()[0] if len(x)>0 else '0',
        'Stock_Coverage_Days':'mean'
        # other columns can be aggregated as needed
    }
    # keep other regressors by taking median if present
    regressors = ['Price_Index','Promotion_Factor','Weather_Factor','Weekend_Factor','Regional_Weight']
    for r in regressors:
        if r in df.columns:
            agg_cols[r] = 'median'

    # perform aggregation
    df_daily = df.groupby(key_cols).agg(agg_cols).reset_index()
    return df_daily

if __name__ == "__main__":
    print("data_preprocessing module - import and use functions in Colab/Script.")

