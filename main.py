from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = FastAPI(title="FlowGuard ML Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schema ────────────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    transaction_date: str   # "YYYY-MM-DD"
    type: str               # "INCOME" | "EXPENSE"
    amount: float

class PredictRequest(BaseModel):
    transactions: List[Transaction]
    horizon: Optional[int] = 90   # jumlah hari prediksi

class DailyForecast(BaseModel):
    forecast_date: str
    predicted_income: float
    predicted_expense: float
    predicted_balance: float
    income_upper: float
    income_lower: float
    expense_upper: float
    expense_lower: float
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


# ── Helper ────────────────────────────────────────────────────────────────────

def build_daily_df(transactions: List[Transaction], tx_type: str) -> pd.DataFrame:
    """Agregasi transaksi per hari untuk satu type (INCOME/EXPENSE)."""
    rows = [
        {"ds": t.transaction_date, "y": t.amount}
        for t in transactions if t.type == tx_type
    ]
    if not rows:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.groupby("ds", as_index=False)["y"].sum()
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def fill_date_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Isi tanggal yang kosong dengan 0 supaya Prophet punya series lengkap."""
    if df.empty:
        return df
    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_range, fill_value=0).reset_index()
    df.columns = ["ds", "y"]
    return df


def fit_prophet(df: pd.DataFrame) -> Prophet:
    """Fit Prophet model. Kalau data < 2 titik, pakai model flat sederhana."""
    model = Prophet(
        interval_width=0.80,         # 80% confidence interval
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=len(df) >= 365,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    return model


def forecast_series(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Return DataFrame dengan kolom: ds, yhat, yhat_lower, yhat_upper
    Kalau data terlalu sedikit, fallback ke rata-rata sederhana.
    """
    if df.empty or len(df) < 2:
        # Fallback: forecast flat dengan nilai 0
        today = datetime.today()
        dates = [today + timedelta(days=i+1) for i in range(horizon)]
        return pd.DataFrame({
            "ds":         dates,
            "yhat":       [0.0] * horizon,
            "yhat_lower": [0.0] * horizon,
            "yhat_upper": [0.0] * horizon,
        })

    df_filled = fill_date_gaps(df)

    if len(df_filled) < 5:
        # Data terlalu sedikit — pakai rata-rata + std sebagai interval
        mean_val = df_filled["y"].mean()
        std_val  = df_filled["y"].std(ddof=0) if len(df_filled) > 1 else mean_val * 0.1
        today    = datetime.today()
        dates    = [today + timedelta(days=i+1) for i in range(horizon)]
        return pd.DataFrame({
            "ds":         dates,
            "yhat":       [max(0, mean_val)] * horizon,
            "yhat_lower": [max(0, mean_val - std_val)] * horizon,
            "yhat_upper": [mean_val + std_val] * horizon,
        })

    model  = fit_prophet(df_filled)
    future = model.make_future_dataframe(periods=horizon)
    fc     = model.predict(future)

    # Ambil hanya baris masa depan
    last_historical_date = df_filled["ds"].max()
    fc_future = fc[fc["ds"] > last_historical_date][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    # Clamp ke non-negatif (income/expense tidak mungkin minus)
    fc_future["yhat"]       = fc_future["yhat"].clip(lower=0)
    fc_future["yhat_lower"] = fc_future["yhat_lower"].clip(lower=0)
    fc_future["yhat_upper"] = fc_future["yhat_upper"].clip(lower=0)

    return fc_future.reset_index(drop=True)


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "FlowGuard ML Forecast API"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.transactions) < 2:
        raise HTTPException(
            status_code=400,
            detail="Minimal 2 transaksi diperlukan untuk generate forecast."
        )

    horizon = max(1, min(req.horizon, 365))

    # Build historical series per type
    income_df  = build_daily_df(req.transactions, "INCOME")
    expense_df = build_daily_df(req.transactions, "EXPENSE")

    # Forecast masing-masing
    income_fc  = forecast_series(income_df,  horizon)
    expense_fc = forecast_series(expense_df, horizon)

    # Gabungkan per tanggal
    details: List[DailyForecast] = []
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
        running_balance_upper += inc_upper - exp_lower   # skenario optimis
        running_balance_lower += inc_lower - exp_upper   # skenario pesimis

        total_income  += inc_val
        total_expense += exp_val

        date_str = income_fc.iloc[i]["ds"].strftime("%Y-%m-%d")

        details.append(DailyForecast(
            forecast_date     = date_str,
            predicted_income  = round(inc_val,   2),
            predicted_expense = round(exp_val,   2),
            predicted_balance = round(running_balance, 2),
            income_upper      = round(inc_upper, 2),
            income_lower      = round(inc_lower, 2),
            expense_upper     = round(exp_upper, 2),
            expense_lower     = round(exp_lower, 2),
            balance_upper     = round(running_balance_upper, 2),
            balance_lower     = round(running_balance_lower, 2),
        ))

    net = total_income - total_expense

    return PredictResponse(
        success                = True,
        horizon                = horizon,
        total_projected_income = round(total_income,  2),
        total_projected_expense= round(total_expense, 2),
        net_cashflow           = round(net, 2),
        net_cashflow_upper     = round(running_balance_upper, 2),
        net_cashflow_lower     = round(running_balance_lower, 2),
        forecast_details       = details,
        model_info             = {
            "model":        "Prophet",
            "data_points":  len(req.transactions),
            "income_days":  len(income_df),
            "expense_days": len(expense_df),
            "horizon_days": horizon,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
    )
