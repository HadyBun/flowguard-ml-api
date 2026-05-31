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


def forecast_series(df: pd.DataFrame, horizon: int):
    today = datetime.today()
    dates = [today + timedelta(days=i+1) for i in range(horizon)]

    if df.empty or len(df) < 2:
        return pd.DataFrame({
            "ds": dates,
            "yhat": [0.0] * horizon,
            "yhat_lower": [0.0] * horizon,
            "yhat_upper": [0.0] * horizon,
        })

    # Isi gap tanggal dengan 0
    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df_filled = df.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    df_filled.columns = ["ds", "y"]

    mean_val = float(df_filled["y"].mean())
    std_val  = float(df_filled["y"].std(ddof=0)) if len(df_filled) > 1 else mean_val * 0.1

    # Hitung trend sederhana pakai linear regression
    x = np.arange(len(df_filled))
    y = df_filled["y"].values
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs[0], coeffs[1]
    else:
        slope, intercept = 0, mean_val

    last_x = len(df_filled)
    yhats, lowers, uppers = [], [], []

    for i in range(horizon):
        xi    = last_x + i
        yhat  = max(0, slope * xi + intercept)
        # Interval 80% menggunakan std dari historical
        lower = max(0, yhat - 1.28 * std_val)
        upper = yhat + 1.28 * std_val
        yhats.append(round(yhat, 2))
        lowers.append(round(lower, 2))
        uppers.append(round(upper, 2))

    return pd.DataFrame({
        "ds":         dates,
        "yhat":       yhats,
        "yhat_lower": lowers,
        "yhat_upper": uppers,
    })


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

    income_fc  = forecast_series(income_df,  horizon)
    expense_fc = forecast_series(expense_df, horizon)

    details = []
    running_balance       = 0.0
    running_balance_upper = 0.0
    running_balance_lower = 0.0
    total_income  = 0.0
    total_expense = 0.0

    n = min(len(income_fc), len(expense_fc), horizon)

    for i in range(n):
        inc_val   = float(income_fc.iloc[i]["yhat"])
        inc_upper = float(income_fc.iloc[i]["yhat_upper"])
        inc_lower = float(income_fc.iloc[i]["yhat_lower"])
        exp_val   = float(expense_fc.iloc[i]["yhat"])
        exp_upper = float(expense_fc.iloc[i]["yhat_upper"])
        exp_lower = float(expense_fc.iloc[i]["yhat_lower"])

        running_balance       += inc_val   - exp_val
        running_balance_upper += inc_upper - exp_lower
        running_balance_lower += inc_lower - exp_upper

        total_income  += inc_val
        total_expense += exp_val

        date_str = income_fc.iloc[i]["ds"].strftime("%Y-%m-%d")

        details.append(DailyForecast(
            forecast_date     = date_str,
            predicted_income  = round(inc_val,   2),
            predicted_expense = round(exp_val,   2),
            predicted_balance = round(running_balance, 2),
            balance_upper     = round(running_balance_upper, 2),
            balance_lower     = round(running_balance_lower, 2),
        ))

    net = total_income - total_expense

    return PredictResponse(
        success                 = True,
        horizon                 = horizon,
        total_projected_income  = round(total_income,  2),
        total_projected_expense = round(total_expense, 2),
        net_cashflow            = round(net, 2),
        net_cashflow_upper      = round(running_balance_upper, 2),
        net_cashflow_lower      = round(running_balance_lower, 2),
        forecast_details        = details,
        model_info              = {
            "model":        "Linear Trend + Std Interval",
            "data_points":  len(req.transactions),
            "income_days":  len(income_df),
            "expense_days": len(expense_df),
            "horizon_days": horizon,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    )
