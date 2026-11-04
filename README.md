# Q-Cast

**Real-time demand forecasting for quick-commerce.**

Q-Cast is an end-to-end machine learning pipeline that forecasts product demand on the M5 Forecasting dataset, built to simulate a 10-minute quick-commerce environment of the Zepto, Blinkit and Swiggy Instamart kind. It takes raw retail sales history and produces per-item daily demand forecasts, with uncertainty, anomaly flags and a business-impact simulation.

> MSc Data Science group project, 2025 to 26. "Predicting what 10-minute delivery customers want, before they order it."

**Topics:** `time-series` `demand-forecasting` `lightgbm` `lstm` `temporal-fusion-transformer` `deep-learning` `m5-forecasting` `quick-commerce` `python`

---

## What it does

The notebook runs the full forecasting workflow in twelve sections: data loading, exploratory analysis, preprocessing, feature engineering, three forecasting models, an ensemble, anomaly detection, evaluation, visualisation, and a business-impact simulation.

Three models are trained and then blended:

| Model | Role | Why it is here |
|---|---|---|
| **LightGBM** | Primary | The workhorse of the M5 competition: fast, handles tabular time-series features natively, SHAP-interpretable. |
| **LSTM** | Deep learning | Learns sequential patterns that trees miss, such as gradual trend shifts over a 28-day lookback. |
| **Temporal Fusion Transformer** | State of the art | Variable selection, attention over long ranges, and quantile (P10, P50, P90) outputs for uncertainty-aware forecasts. |
| **Ensemble** | Combined | Inverse-MAE weighted average of the above, which reduces variance and generalises better. |

## Dataset

The pipeline uses the [M5 Forecasting Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) data from Kaggle. Download and place these in a `data/` folder:

- `sales_train_evaluation.csv` (daily unit sales for 30,490 products over 1,941 days)
- `calendar.csv` (weekday, month, SNAP flags, special events)
- `sell_prices.csv` (weekly selling price per store and item)

A random sample of items is used for faster runs while keeping all three M5 categories (Foods, Hobbies, Household) represented.

## Feature engineering

The most important section of the pipeline. Features are grouped by the signal they capture:

| Group | Features | Rationale |
|---|---|---|
| Lag | lag_1, lag_7, lag_14, lag_28 | Yesterday, same day last week, fortnight and month |
| Rolling | rolling 7, 14, 28 mean and std | Local trend and volatility |
| Price | sell_price, price_change | Promotional and elasticity signal |
| Temporal | dayofweek, month, year, weekofyear, quarter | Seasonality at multiple scales |
| Calendar | is_event, is_snap, event encoding | Demand spikes on event and benefit days |
| Fourier | sin and cos at weekly and yearly frequency | Smooth seasonality representation |

## Anomaly detection

An Isolation Forest flags demand anomalies, which in quick-commerce map onto real events: IPL match nights lifting snacks and beverages, rain lifting hot food, supply shocks dropping demand to zero, and flash sales or app promotions causing abnormal uplift.

## Evaluation

All models are compared on four metrics, with a time-based split that holds out the last 28 days to mirror the M5 evaluation protocol.

| Metric | Reads as |
|---|---|
| MAE | Average absolute error, same units as sales |
| RMSE | Penalises large errors, which matters for spike events |
| MAPE | Relative percentage error, scale-independent |
| SMAPE | Symmetric MAPE, the M5 competition standard |

A closing business-impact simulation runs a simple inventory policy over the test window, stocking `forecast * (1 + safety_factor)` each day, and measures stockouts against waste to put a value on better forecasts.

## Files

```
QCast_Demand_Forecasting (1).ipynb   The full pipeline notebook (run this)
QCast_Demand_Forecasting (1).html    Rendered notebook with all outputs and charts
QCast_Demand_Forecasting (1).pdf     PDF export of the notebook
Q-Cast_Project_Report.docx           Written project report
```

## How to run

1. Install the dependencies listed at the top of the notebook: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `scikit-learn`, `lightgbm`, `torch`, `pytorch-forecasting`, `pytorch-lightning`, `ipywidgets`.
2. Download the three M5 CSV files into a `data/` folder next to the notebook.
3. Open `QCast_Demand_Forecasting (1).ipynb` and run the cells top to bottom.

To read the results without running anything, open the rendered `QCast_Demand_Forecasting (1).html`, which keeps every chart and table inline.
