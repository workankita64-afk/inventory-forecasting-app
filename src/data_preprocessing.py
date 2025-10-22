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
    """
    Cleans the raw dataframe by:
    - Parsing dates correctly (handles multiple date formats, day-first)
    - Ensuring Daily_Sales is numeric
    - Dropping rows with missing essential values
    """
    print("Cleaning data...")
    
    # ✅ Correct date parsing
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format='mixed')
    
    # Keep only valid dates
    df = df.dropna(subset=["Date"])
    
    # Convert Daily_Sales to numeric
    df["Daily_Sales"] = pd.to_numeric(df["Daily_Sales"], errors="coerce")
    df = df.dropna(subset=["Daily_Sales"])
    
    # Remove negative or impossible sales
    df = df[df["Daily_Sales"] >= 0]
    
    print("✅ Data cleaned successfully.")
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

