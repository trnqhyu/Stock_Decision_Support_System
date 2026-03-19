import streamlit as st
import pandas as pd
import requests
from datetime import date

st.set_page_config(
    page_title="Lọc cổ phiếu",
    layout="wide"
)

st.title("Lọc cổ phiếu")


# ===== CONFIG =====
BACKEND_URL = "http://127.0.0.1:8000/predict"

# ===== LOAD DATA=====
@st.cache_data
def load_data():
    return pd.read_csv("backend/data/stock_ml_dataset.csv", parse_dates=["date"])

df = load_data()

# ===== UI =====
st.sidebar.header("Điều kiện lọc")

selected_date = st.sidebar.date_input(
    "Ngày phân tích",
    value=df["date"].max().date(),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date()
)

decisions = st.sidebar.multiselect(
    "Loại khuyến nghị",
    ["BUY", "HOLD", "SELL"],
    default=["BUY", "HOLD"]
)

min_conf = st.sidebar.slider(
    "Độ tin tưởng tối thiểu",
    min_value=0.5,
    max_value=0.9,
    value=0.7,
    step=0.05
)

run_filter = st.sidebar.button("Lọc cổ phiếu")

# ===== LOGIC =====
if run_filter:
    results = []

    tickers = df["ticker"].unique()

    with st.spinner("Đang phân tích danh sách cổ phiếu..."):
        for ticker in tickers:
            payload = {
                "ticker": ticker,
                "date": selected_date.strftime("%Y-%m-%d")
            }

            try:
                r = requests.post(BACKEND_URL, json=payload, timeout=2)
                if r.status_code == 200:
                    res = r.json()
                    if (
                        res["decision"] in decisions
                        and res["confidence"] >= min_conf
                    ):
                        results.append(res)
            except:
                continue

    if results:
        result_df = pd.DataFrame(results)
        price_df = pd.json_normalize(result_df["price"])
        price_df.columns = ["open", "high", "low", "close"]

        result_df = pd.concat(
         [result_df.drop(columns=["price"]), price_df],
         axis=1
        )

        result_df = result_df.sort_values("confidence", ascending=False)

        st.success(f" Tìm thấy {len(result_df)} cổ phiếu phù hợp")
        st.dataframe(
          result_df.reset_index(drop=True),
          use_container_width=True
        )


        