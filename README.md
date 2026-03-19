# Stock_Decision_Support_System
Hệ thống hỗ trợ quyết định giao dịch cổ phiếu gồm:

- **Backend (FastAPI)**: cung cấp API dự đoán khuyến nghị **BUY / HOLD / SELL** từ mô hình ML.
- **Frontend (Streamlit)**: giao diện chọn mã cổ phiếu + ngày, hiển thị khuyến nghị, độ tin tưởng, biểu đồ giá, và xuất Excel danh sách phân tích.

## Yêu cầu

- **Windows 10/11**
- **Python 3.10+** (khuyến nghị 3.11)
- (Khuyến nghị) tạo môi trường ảo `venv`

## Cấu trúc thư mục chính

```
PTCK/
├─ backend/
│  ├─ main.py
│  ├─ requirements.txt
│  └─ data/
│     └─ stock_ml_dataset.csv
├─ Hệ_thống.py
└─ README.md
```

## Dữ liệu & mô hình

- **Dataset**: `backend/data/stock_ml_dataset.csv`
- **Model file (bắt buộc để backend dự đoán)**:
  - `backend/main.py` đang load tại: `backend/model/xgboost_final_stock_model.pkl`
  - Nếu máy bạn chưa có file này, hãy tạo thư mục `backend/model/` và đặt đúng tên file `.pkl` vào đó.

## Cài đặt

Mở PowerShell tại thư mục `c:\PTCK`.

### 1) Tạo và kích hoạt môi trường ảo (khuyến nghị)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Cài thư viện backend

```powershell
pip install -r .\backend\requirements.txt
```

### 3) Cài thư viện frontend (Streamlit UI)

`Hệ_thống.py` dùng thêm `streamlit`, `requests`, `matplotlib`, và export Excel (pandas cần engine).

```powershell
pip install streamlit requests matplotlib openpyxl
```

## Chạy dự án

### 1) Chạy backend (FastAPI)

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- Endpoint dự đoán: `POST http://127.0.0.1:8000/predict`

### 2) Chạy frontend (Streamlit)

Mở **terminal mới** (vẫn trong `c:\PTCK`, và vẫn đang activate `.venv` nếu dùng).

```powershell
streamlit run ".\Hệ_thống.py"
```

Frontend sẽ gọi backend theo biến:

- `BACKEND_URL = "http://127.0.0.1:8000/predict"`

Nếu backend chưa chạy, UI sẽ hiển thị **demo** (có cảnh báo) và vẫn cho xem biểu đồ/dữ liệu.

## API `/predict`

### Request body

```json
{
  "ticker": "ABC",
  "date": "2025-11-25"
}
```

### Response (ví dụ)

```json
{
  "ticker": "ABC",
  "used_date": "2025-11-25",
  "price": { "open": 10.5, "high": 10.8, "low": 10.2, "close": 10.6 },
  "decision": "HOLD",
  "confidence": 0.732
}
```


