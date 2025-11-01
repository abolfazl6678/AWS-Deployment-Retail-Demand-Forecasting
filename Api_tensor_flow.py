from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import date
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Demand Forecast for Retail Stores Chain",
    version="1.0",
    description="Enter required information to get daily demand forecast for list of products/stores based on developed deep learning model."
)

# ---------------------------
# Input and Output Schemas
# ---------------------------
class InputData(BaseModel):
    Date: date = Field(..., description="Date in YYYY-MM-DD format")
    Store_ID: str = Field(..., description="Store code : S001, S002, S003, S004, S005")
    Product_ID: str = Field(..., description="Product code : P0001, P0002, ... P0020")
    Category: str = Field(..., description="Product category : Clothing, Electronics, Furniture, Groceries, Toys")
    Region: str = Field(..., description="Store region : East, North, South, West")
    Inventory_Level: int
    Units_Sold: int
    Units_Ordered: int
    Price: float
    Discount: int
    Weather_Condition: str = Field(..., description="Weather condition : Cloudy, Rainy, Snowy, Sunny")
    Holiday_Promotion: int
    Competitor_Pricing: float
    Seasonality: str = Field(..., description="Season : Autumn, Spring, Summer, Winter")


class OutputData(BaseModel):
    prediction: float


# ---------------------------
# Load Pretrained Objects
# ---------------------------
onh = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("standard_scaler.pkl")
model = load_model("DL_model_tf.keras")


# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict", response_model=OutputData)
def demand_forecast(data: InputData):
    # Categorical features for one-hot encoding
    categorical_features = [
        data.Store_ID,
        data.Product_ID,
        data.Category,
        data.Region,
        data.Weather_Condition,
        data.Seasonality
    ]

    # Numerical features for scaling
    numerical_features = [
        data.Inventory_Level,
        data.Units_Sold,
        data.Units_Ordered,
        data.Price,
        data.Discount,
        data.Holiday_Promotion,
        data.Competitor_Pricing
    ]

    # Date features
    dayofweek = data.Date.weekday()
    dayofweek_sin = np.sin(2 * np.pi * dayofweek / 7).reshape(-1, 1)
    dayofweek_cos = np.cos(2 * np.pi * dayofweek / 7).reshape(-1, 1)

    # Process categorical & numerical
    X_cat_encoded = onh.transform([categorical_features]).toarray()
    X_num = np.array([numerical_features])

    # Combine all features
    X_combined = np.hstack((X_num, X_cat_encoded, dayofweek_sin, dayofweek_cos))
    X_processed = scaler.transform(X_combined)

    # Predict
    pred = model.predict(X_processed)[0][0]

    # Round and format to exactly 2 decimals
    rounded_result = float(f"{pred:.2f}")

    return OutputData(prediction=rounded_result)
# ---------------------------
# Health Check
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "Retail Demand Forecasting API"}


@app.get("/health", summary="API health check", description="Check if all model artifacts are loaded and API is ready")
def health_check():
    return {
        "status": "healthy" if (onh is not None and scaler is not None) else "degraded",
        "encoders_loaded": onh is not None,
        "scalers_loaded": scaler is not None
    }
