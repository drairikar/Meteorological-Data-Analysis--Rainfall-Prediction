# Rainfall Forecasting with AR and ARIMA Models

A time-series study of one year of hourly meteorological data. It covers stationarity testing, ACF/PACF
analysis, and AR/ARIMA forecasts of rainfall at **daily** and **hourly** horizons. Course project for
**EE4C03 Statistical Digital Signal Processing and Modeling**, TU Delft.

## Data

`Climate344.csv` holds 8 784 hourly records (one leap year) of irradiance, ambient and ground temperature,
wind, cloud cover, pressure, rain (mm/h) and solar geometry. Rain is summed into daily, weekly and monthly
series.

## Method

1. **Exploration**: resampled rainfall series, rolling mean and standard deviation.
2. **Stationarity**: augmented Dickey–Fuller tests. Daily and hourly series are stationary (p < 0.001);
   weekly and monthly series are differenced first.
3. **Order selection**: ACF/PACF up to lag 100.
4. **Models**: train up to 10 Sept, test from 11 Sept to 31 Dec.
   - Daily: ARIMA(90, 0, 1)
   - Hourly: AR(900) via `statsmodels.AutoReg`

## Results, compared with naive baselines

| Horizon | Model | MAE | RMSE |
| --- | --- | ---: | ---: |
| Daily (112 test days, mm/day) | ARIMA(90, 0, 1) | 0.803 | 1.049 |
| | Baseline: training mean | 0.798 | 1.019 |
| | Baseline: persistence (yesterday's value) | **0.752** | **0.985** |
| Hourly (last 100 test hours, mm/h) | AR(900) | 0.074 | 0.089 |
| | Baseline: training mean | **0.066** | **0.088** |

Model scores come from the notebook; baseline scores from `python baselines.py`, which uses the same
split.

**Takeaway:** neither high-order model beats a naive forecast. Rainfall is intermittent and heavy-tailed,
which breaks the Gaussian assumptions behind linear AR/ARIMA. With hundreds of lags the models mostly fit
noise, and their multi-step forecasts decay back to the mean. Next steps that would be worth trying:

- A two-part model: rain/no-rain classification, then intensity regression.
- Exogenous predictors (pressure tendency, cloud cover, humidity) via SARIMAX or gradient boosting.
- Rolling-origin evaluation with one-step-ahead forecasts instead of one long multi-step forecast.

## Running

```bash
pip install -r requirements.txt
jupyter notebook SDSP_Rainfall_Analysis.ipynb
python baselines.py
```
