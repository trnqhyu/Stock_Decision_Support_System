import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import date
import requests
from requests.exceptions import RequestException
from datetime import date
from datetime import datetime

# ================= CONFIG =================
BACKEND_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Stock Decision Support System",
    layout="wide"
)

st.title("Stock Decision Support System")

# ================= LOAD DATA =================
import os
import pandas as pd
import streamlit as st

# ===== SESSION STATE =====
if "analysis_list" not in st.session_state:
    st.session_state.analysis_list = []

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "backend", "data", "stock_ml_dataset.csv")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ================= SIDEBAR =================
st.sidebar.header("Chọn thông tin")

ticker = st.sidebar.selectbox(
    "Mã cổ phiếu",
    sorted(df["ticker"].unique())
)

selected_date = st.sidebar.date_input(
    "Ngày phân tích",
    value=date(2025, 11, 25),
    min_value=df["date"].min().date(),
    max_value=df["date"].max().date()
)

# ================= CALL BACKEND =================
payload = {
    "ticker": ticker,
    "date": selected_date.strftime("%Y-%m-%d")
}

result = None
backend_error = None

try:
    response = requests.post(BACKEND_URL, json=payload, timeout=3)
    if response.status_code == 200:
        result = response.json()
    else:
        backend_error = f"Backend error: {response.status_code}"
except RequestException as e:
    backend_error = str(e)

if result is None:
    st.warning("⚠️ Backend chưa chạy. Hiển thị dữ liệu demo.")

    result = {
        "ticker": ticker,
        "used_date": str(selected_date),
        "decision": "HOLD",
        "confidence": 0.50,
        "price": {
            "open": None,
            "high": None,
            "low": None,
            "close": None
        }
    }

# ================= DISPLAY RESULT =================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📌 Khuyến nghị",
        value=result["decision"]
    )

with col2:
    st.metric(
        label="🎯 Độ tin tưởng",
        value=result["confidence"]
    )

with col3:
    display_date = datetime.strptime(
    result["used_date"], "%Y-%m-%d"
     ).strftime("%d/%m/%Y")

    st.metric("📅 Ngày sử dụng", display_date)
# ================= PRICE CHART =================
st.subheader("📊 Biểu đồ giá")

chart_df = df[df["ticker"] == ticker].sort_values("date")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(chart_df["date"], chart_df["close"], label="Giá đóng cửa")
ax.axvline(pd.to_datetime(result["used_date"]), color="red", linestyle="--", label="Ngày")

ax.set_xlabel("Thời gian")
ax.set_ylabel("Giá")
ax.legend()

st.pyplot(fig)

# ===== CHUẨN HÓA DỮ LIỆU SAU KHI CÓ RESULT =====
decision = result["decision"]
confidence = result["confidence"]
price = result.get("price", {})

# Ngày hiển thị DD/MM/YYYY
display_date = selected_date.strftime("%d/%m/%Y")

# ================= PRICE INFO =================
st.subheader("💰 Giá cổ phiếu")

price_df = pd.DataFrame({
    "Open": [price.get("open")],
    "High": [price.get("high")],
    "Low": [price.get("low")],
    "Close": [price.get("close")]
}, index=[ticker])

price_df.index.name = "Mã cổ phiếu"

st.table(price_df)


if st.button("➕ Thêm vào danh sách phân tích"):
    st.session_state.analysis_list.append({
        "Mã CP": ticker,
        "Ngày phân tích": display_date,
        "Khuyến nghị": decision,
        "Độ tin tưởng": confidence,
        "Open": price.get("open"),
        "High": price.get("high"),
        "Low": price.get("low"),
        "Close": price.get("close")
    })
    st.success("Đã thêm vào danh sách")
# ================= Chỉ báo =================
df["date"] = pd.to_datetime(df["date"])

df_sel = df[
    (df["ticker"] == ticker) &
    (df["date"] <= pd.to_datetime(selected_date))
].sort_values("date")

if df_sel.empty:
    st.warning("⚠️ Không có dữ liệu phù hợp cho ngày đã chọn")
    st.stop()

row = df_sel.iloc[-1]

# Ngưỡng volatility thấp (30% thấp nhất)
vol_threshold = df_sel["volatility_10d"].quantile(0.3)

# Tính volume trung bình 20 ngày
df = df.sort_values(["ticker", "date"])

df["volume_ma20"] = (
    df
    .groupby("ticker")["volume"]
    .transform(lambda x: x.rolling(window=20, min_periods=1).mean())
)
# ================= RULE VIỆT NAM (POST-PROCESSING) =================

def rule_volume_dry_up(row):
    # Cạn cung: volume thấp hơn trung bình
    if "volume" not in row or "volume_ma20" not in row:
        return False
    return row["volume"] < 0.6 * row["volume_ma20"]


def rule_low_volatility(row, vol_threshold):
    # Nén giá
    return row["volatility_10d"] < vol_threshold


def rule_no_sell_at_bottom(row, vol_threshold):
    # RSI rất thấp + vol thấp → không bán đáy
    return row["rsi_14"] < 25 and row["volatility_10d"] < vol_threshold

def rule_above_key_ma(row):
    # Giá vẫn trên MA20 & MA50 → chưa gãy xu hướng
    return (
        row["close"] > row["ma_20"] and
        row["close"] > row["ma_50"]
    )

def rule_pullback_not_break_trend(row):
    # Điều chỉnh kỹ thuật trong xu hướng tăng
    return (
        row["trend_slope_20d"] > 0 and
        row["return_1d"] < 0 and
        row["return_5d"] > 0
    )

def rule_price_down_no_volume(row):
    # Giảm giá nhưng không có volume xác nhận
    return (
        row["return_1d"] < 0 and
        row["volume"] < row["volume_ma20"]
    )

def rule_rsi_rebound(row):
    # RSI vừa thoát quá bán → có thể hồi kỹ thuật
    return (
        row["rsi_14"] > 30 and
        row["return_1d"] > 0
    )

def rule_rsi_overbought(row):
    # RSI quá cao → tránh mua đuổi
    return row["rsi_14"] >= 70

def rule_price_too_far_from_ma(row, threshold=0.08):
    # Giá đã xa MA → rủi ro điều chỉnh
    return (
        (row["close"] - row["ma_20"]) / row["ma_20"] > threshold or
        (row["close"] - row["ma_50"]) / row["ma_50"] > threshold
    )

def rule_high_volatility(row, vol_threshold):
    # Biến động cao → không vào lệnh
    return row["volatility_10d"] > vol_threshold

def rule_price_up_no_volume(row):
    # Giá tăng không có volume xác nhận
    return (
        row["return_1d"] > 0 and
        row["volume"] < row["volume_ma20"]
    )
# ================= APPLY RULE VIỆT NAM =================

final_decision = decision
final_reason = "các chỉ báo không có gì bất thường"

# Chỉ can thiệp khi ML khuyến nghị SELL
if decision == "SELL" and confidence >= 0.6:

    if rule_no_sell_at_bottom(row, vol_threshold):
        final_decision = "HOLD"
        final_reason = "RSI rất thấp + biến động thấp → tránh bán đáy"

    elif rule_volume_dry_up(row):
        final_decision = "HOLD"
        final_reason = "Thanh khoản cạn (cạn cung) → không bán"

    elif rule_low_volatility(row, vol_threshold):
        final_decision = "HOLD"
        final_reason = "Giá đang nén, thị trường chờ tín hiệu"

    elif rule_above_key_ma(row):
        final_decision = "HOLD"
        final_reason = "Giá vẫn trên MA20 & MA50 → chưa gãy xu hướng"

    elif rule_pullback_not_break_trend(row):
        final_decision = "HOLD"
        final_reason = "Điều chỉnh kỹ thuật trong xu hướng tăng"

    elif rule_price_down_no_volume(row):
        final_decision = "HOLD"
        final_reason = "Giảm giá nhưng không có volume xác nhận"

    elif rule_rsi_rebound(row):
        final_decision = "HOLD"
        final_reason = "RSI vừa thoát quá bán → có thể hồi kỹ thuật"


if decision == "BUY" and confidence >= 0.6:

    if rule_rsi_overbought(row):
        final_decision = "HOLD"
        final_reason = "RSI quá cao → tránh mua đuổi"

    elif rule_price_too_far_from_ma(row):
        final_decision = "HOLD"
        final_reason = "Giá đã xa MA → rủi ro điều chỉnh"

    elif rule_high_volatility(row, vol_threshold):
        final_decision = "HOLD"
        final_reason = "Biến động cao → không vào lệnh"

    elif rule_price_up_no_volume(row):
        final_decision = "HOLD"
        final_reason = "Giá tăng không có volume xác nhận"


st.subheader("📄 Các chỉ báo hỗ trợ quyết định")

with st.container():

    col1, col2 = st.columns(2)

    # ===== RSI =====
    with col1:
        rsi = row["rsi_14"]
        if rsi > 70:
            rsi_status = "🔴 Quá mua"
        elif rsi < 30:
            rsi_status = "🟢 Quá bán"
        else:
            rsi_status = "🟡 Trung tính"

        st.metric("RSI (14)", round(rsi, 2))
        st.caption(rsi_status)


    # ===== VOLATILITY =====
    with col2:
        vol = row["volatility_10d"]
        st.metric("Volatility (10d)", round(vol, 4))

        if vol > 0.03:
            st.caption("⚠️ Biến động cao")
        else:
            st.caption("✅ Biến động thấp")

st.caption(f"Xem xét {final_decision} vì {final_reason}")




# ================= Danh sách phan tich =================

st.divider()
st.subheader("📋 Danh sách phân tích")

if st.session_state.analysis_list:
    df_list = pd.DataFrame(st.session_state.analysis_list)
    st.dataframe(df_list, use_container_width=True)
else:
    st.info("Chưa có dữ liệu trong danh sách")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Tạo danh sách mới"):
        st.session_state.analysis_list = []
        st.success("Đã reset danh sách")

with col2:
    st.write("")  # để cân layout

# ================= EXPORT EXCEL =================
st.subheader("📥 Xuất kết quả")

export_df = pd.DataFrame([{
    "Ticker": ticker,
    "Date": result["used_date"],
    "Decision": result["decision"],
    "Confidence": result["confidence"],
    "Open": result["price"]["open"],
    "High": result["price"]["high"],
    "Low": result["price"]["low"],
    "Close": result["price"]["close"]
}])

import io

if st.session_state.analysis_list:
    buffer = io.BytesIO()
    df_list.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        label="📥 Xuất danh sách ra Excel",
        data=buffer,
        file_name="stock_decision_list.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


