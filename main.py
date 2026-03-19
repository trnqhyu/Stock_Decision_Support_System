from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# ========== LOAD MODEL ==========
model = joblib.load("backend/model/xgboost_final_stock_model.pkl")

# ========== LOAD DATA ==========
data = pd.read_csv("backend/data/stock_ml_dataset.csv")
data["date"] = pd.to_datetime(data["date"])

FEATURES = [
    # Momentum
    "rsi_14",
    "return_1d",
    "return_5d",

    # Volatility
    "volatility_10d",

    # Trend structure
    "ma_diff",
    "trend_slope_20d",

    # Market context
    "wyckoff_kmeans",

]

LABEL_MAP = {
    0: "SELL",
    1: "HOLD",
    2: "BUY"
}

# ========== FASTAPI ==========
app = FastAPI(title="Stock Decision Support System")

class PredictRequest(BaseModel):
    ticker: str
    date: str   # YYYY-MM-DD


from datetime import datetime

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        query_date = pd.to_datetime(req.date)

        # Lấy phiên giao dịch gần nhất <= ngày yêu cầu
        df = data[
            (data["ticker"] == req.ticker) &
            (data["date"] <= query_date)
        ].sort_values("date", ascending=False)

        if df.empty:
            return {
                "error": "No data available for this ticker"
            }

        row = df.iloc[0]

        # ===== LẤY GIÁ =====
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        # ===== ML =====
        X = row[FEATURES].values.reshape(1, -1)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X).max()

        return {
            "ticker": req.ticker,
            "used_date": row["date"].strftime("%Y-%m-%d"),

            "price": {
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2)
            },

            "decision": LABEL_MAP[pred],
            "confidence": round(float(proba), 3)
        }

    except Exception as e:
        return {
            "error": str(e)
        }


