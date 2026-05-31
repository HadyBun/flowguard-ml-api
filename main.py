from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

app = FastAPI(title="FlowGuard ML Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaction(BaseModel):
    transaction_date: str
    type: str
    amount: float

class PredictRequest(BaseModel):
    transactions: List[Transaction]
    horizon: Optional[int] = 90

class DailyForecast(BaseModel):
    forecast_date: str
    predicted_income: float
    predicted_expense: float
    predicted_balance: float
    balance_upper: float
    balance_lower: float

class PredictResponse(BaseModel):
    success: bool
    horizon: int
    total_projected_income: float
    total_projected_expense: float
    net_cashflow: float
    net_cashflow_upper: float
    net_cashflow_lower: float
    forecast_details: List[DailyForecast]
    model_info: dict


def build_daily_df(transactions: List[Transaction], tx_type: str) -> pd.DataFrame:
    rows = [
        {"ds": t.transaction_date, "y": t.amount}
        for t in transactions if t.type == tx_type
    ]
    if not rows:
        return pd.DataFrame(columns=["ds", "y"])
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.groupby("ds", as_index=False)["y"].sum()
    return df.sort_values("ds").reset_index(drop=True)


def get_avg_daily(df: pd.DataFrame) -> tuple:
    """
    Return (avg, std) per hari dari data historis.
    Pakai rata-rata sederhana — lebih stabil untuk data sedikit.
    """
    if df.empty:
        return 0.0, 0.0

    # Isi gap tanggal dengan 0
    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df_filled = df.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    df_filled.columns = ["ds", "y"]

    avg = float(df_filled["y"].mean())
    std = float(df_filled["y"].std(ddof=0)) if len(df_filled) > 1 else avg * 0.15

    return avg, std


@app.get("/")
def root():
    return {"status": "ok", "service": "FlowGuard ML Forecast API"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.transactions) < 2:
        raise HTTPException(
            status_code=400,
            detail="Minimal 2 transaksi diperlukan."
        )

    horizon = max(1, min(req.horizon, 365))

    income_df  = build_daily_df(req.transactions, "INCOME")
    expense_df = build_daily_df(req.transactions, "EXPENSE")

    # Hitung rata-rata & std harian dari historical data
    avg_income,  std_income  = get_avg_daily(income_df)
    avg_expense, std_expense = get_avg_daily(expense_df)

    # Hitung current balance dari semua transaksi historis
    total_historical_income  = sum(t.amount for t in req.transactions if t.type == "INCOME")
    total_historical_expense = sum(t.amount for t in req.transactions if t.type == "EXPENSE")
    current_balance = total_historical_income - total_historical_expense

    today = datetime.today()
    details = []

    running_balance       = current_balance
    running_balance_upper = current_balance
    running_balance_lower = current_balance

    total_projected_income  = 0.0
    total_projected_expense = 0.0

    for i in range(horizon):
        date_str = (today + timedelta(days=i+1)).strftime("%Y-%m-%d")

        # Prediksi harian = rata-rata historis (realistic)
        daily_income  = max(0.0, avg_income)
        daily_expense = max(0.0, avg_expense)

        # Skenario optimistic: income lebih tinggi, expense lebih rendah
        daily_income_upper  = max(0.0, avg_income  + 1.28 * std_income)
        daily_expense_lower = max(0.0, avg_expense - 1.28 * std_expense)

        # Skenario pessimistic: income lebih rendah, expense lebih tinggi
        daily_income_lower  = max(0.0, avg_income  - 1.28 * std_income)
        daily_expense_upper = max(0.0, avg_expense + 1.28 * std_expense)

        running_balance       += daily_income  - daily_expense
        running_balance_upper += daily_income_upper  - daily_expense_lower
        running_balance_lower += daily_income_lower  - daily_expense_upper

        total_projected_income  += daily_income
        total_projected_expense += daily_expense

        details.append(DailyForecast(
            forecast_date     = date_str,
            predicted_income  = round(daily_income,   2),
            predicted_expense = round(daily_expense,  2),
            predicted_balance = round(running_balance, 2),
            balance_upper     = round(running_balance_upper, 2),
            balance_lower     = round(running_balance_lower, 2),
        ))

    net = total_projected_income - total_projected_expense

    return PredictResponse(
        success                 = True,
        horizon                 = horizon,
        total_projected_income  = round(total_projected_income,  2),
        total_projected_expense = round(total_projected_expense, 2),
        net_cashflow            = round(net, 2),
        net_cashflow_upper      = round(running_balance_upper - current_balance, 2),
        net_cashflow_lower      = round(running_balance_lower - current_balance, 2),
        forecast_details        = details,
        model_info              = {
            "model":            "Daily Average + Std Interval",
            "current_balance":  round(current_balance, 2),
            "avg_daily_income": round(avg_income,  2),
            "avg_daily_expense":round(avg_expense, 2),
            "data_points":      len(req.transactions),
            "horizon_days":     horizon,
            "generated_at":     datetime.utcnow().isoformat() + "Z",
        },
    )