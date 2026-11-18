"""Naive baselines for the rainfall forecasts in SDSP_Rainfall_Analysis.ipynb.

Uses the same data, resampling and train/test split as the notebook, so the
numbers can be compared directly with the ARIMA / AR results.
"""
import numpy as np
import pandas as pd


def load_rain(path="Climate344.csv"):
    df = pd.read_csv(path)
    # Timestamps use year 0000; the notebook maps them to 2024.
    idx = pd.to_datetime(df["Local []"].str.replace("0000", "2024"))
    return pd.Series(df["Rain [mm/hr]"].values, index=idx)


def scores(y, yhat):
    err = np.asarray(y) - np.asarray(yhat)
    return np.abs(err).mean(), np.sqrt((err ** 2).mean())


def main():
    rain = load_rain()
    daily = rain.resample("D").sum()
    hourly = rain.resample("h").sum()

    train_d, test_d = daily.loc[:"2024-09-10"], daily.loc["2024-09-11":]
    train_h, test_h = hourly.loc[:"2024-09-10"], hourly.loc["2024-09-11":][-100:]

    rows = [
        ("daily", "train mean", *scores(test_d, np.full(len(test_d), train_d.mean()))),
        ("daily", "persistence (yesterday)", *scores(test_d, daily.shift(1).loc[test_d.index])),
        ("hourly, last 100 h", "train mean", *scores(test_h, np.full(len(test_h), train_h.mean()))),
    ]
    print(f"{'horizon':<20}{'baseline':<26}{'MAE':>8}{'RMSE':>8}")
    for horizon, name, mae, rmse in rows:
        print(f"{horizon:<20}{name:<26}{mae:8.3f}{rmse:8.3f}")


if __name__ == "__main__":
    main()
