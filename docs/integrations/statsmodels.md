---
title: statsmodels integration
---

# Fitting AR and VAR models with statsmodels

pymts includes stochastic processes (AR, VAR, OU, Brownian) ideal for practising with [statsmodels](https://www.statsmodels.org/).

## 1. Generate training data

`python
from pymts import Generator

gen = Generator()
var_cfg = {
    "model": "var",
    "M": 3,
    "T": 200,
    "p": 2,
    "noise_scale": 0.1,
    "seed": 77,
}
var_ds = gen.generate([var_cfg])[0]
`

## 2. Convert to pandas

`python
var_df = var_ds.pymts.to_dataframe()
wide = var_df.pivot_table(index="time", columns="channel", values="value")
wide.columns = [f"channel_{c}" for c in wide.columns]
`

## 3. Fit a VAR model

`python
from statsmodels.tsa.api import VAR

model = VAR(wide)
fit = model.fit(maxlags=2)
print(fit.summary())
`

## 4. Forecast

`python
forecast = fit.forecast(y=wide.values, steps=10)
`

Use it.plot_forecast() or Matplotlib to visualise predictions alongside pymts ground truth.

## 5. AR example

`python
ar_cfg = {"model": "ar", "M": 1, "T": 200, "p": 2, "phi": [0.6, -0.3], "noise_std": 0.05, "seed": 88}
ar_ds = gen.generate([ar_cfg])[0]
ar_series = ar_ds.pymts.to_dataframe()
ar_series = ar_series[ar_series["channel"] == 0].pivot(index="time", columns="realization", values="value")
`

Fit per realization:

`python
from statsmodels.tsa.ar_model import AutoReg

results = [AutoReg(ar_series[col], lags=2).fit() for col in ar_series.columns]
`

Compare the fitted coefficients against pymts phi to validate stability.

## Best practices

- Combine statsmodels diagnostics (ACF/PACF, Ljung-Box) with pymts attrs to understand generated dynamics.
- Use the CLI to persist large cohorts; load them lazily when fitting multiple models.
- Document base seeds so you can reproduce identical simulations for benchmarking.
