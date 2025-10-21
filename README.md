# Inventory Forecasting App (Cloud-Based, Billing-Free)
This application forecasts demand, calculates reorder thresholds, and generates automated inventory recommendations using Google Drive for data storage, Google Colab for computation, and Streamlit Cloud for dashboard hosting.

## Architecture
- Data: Google Drive / Cloud Storage Free Tier
- Model Training: Google Colab Free Tier
- Dashboard: Streamlit Community Cloud (free)
- API: FastAPI hosted on Streamlit/HuggingFace

## Next Steps
1. Upload dataset to Google Drive
2. Run forecasting notebook in Colab
3. Push results to dashboard/API
