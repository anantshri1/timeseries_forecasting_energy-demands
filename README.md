# Energy Demand Forecasting & Climate Scenario Simulation

Forecasting India's national electricity demand using machine learning, weather enrichment, and recursive scenario-based simulation. Data sourced from Grid India, Indian Meteorological Department (IMD), and Regional Load Dispatch Centres (2019-2024) [available here](https://www.kaggle.com/datasets/shubhamvashisht/hourly-load-india-electrical-load-forecasting). 

---
## Overview

Forecasting electricity demand is a core problem in power system planning and operations. Accurate short-term forecasts support generation scheduling, reserve planning, and system reliability. This project develops a 24-hour ahead national electricity demand forecasting and scenario analysis framework using Indian grid demand data.

This project develops a machine learning framework for short-term electricity demand forecasting across India. Starting from historical national and regional load data, the workflow incorporates autoregressive features, weather variables from multiple cities, cyclical temporal encodings, and explainable AI techniques to improve forecast accuracy.

Beyond conventional forecasting, the project extends into scenario-aware modelling by simulating the impact of heatwaves and coldwaves on future electricity demand using recursive multi-step forecasting.

---

## Objectives

* Forecast national electricity demand 24 hours ahead.
* Evaluate the value of regional demand signals.
* Incorporate weather information into load forecasting.
* Quantify the impact of temperature shocks on future demand.
* Build a reproducible experimentation pipeline using MLflow.

---
## Exploratory Data Analysis
The dataset contains hourly electricity demand across India:

* National demand
* Northern region demand
* Southern region demand
* Western region demand
* Eastern region demand
* North-Eastern region demand

Coverage:

* Hourly observations
* Multiple years of historical demand data

Initial analysis shows national hourly demand peaks in the summer and drops to a minimum in the autumn:

<img width="1026" height="537" alt="Screenshot 2026-06-02 at 3 17 50 PM" src="https://github.com/user-attachments/assets/76377641-fce5-43ee-aebf-6bf3c4424324" />

with monthly and hourly trends shown below:

<img width="1104" height="675" alt="Screenshot 2026-06-02 at 3 18 49 PM" src="https://github.com/user-attachments/assets/056428c4-1bfb-4899-9806-aa30e9ee4e61" />

<img width="1106" height="672" alt="Screenshot 2026-06-02 at 3 19 10 PM" src="https://github.com/user-attachments/assets/e6b22414-d73b-46d3-9551-2000f685e7ca" />

Regional demand exhibits two distinct behavioural clusters. The Northern, Eastern and North-Eastern regions show pronounced evening demand peaks (~19:00), while the Southern and Western regions peak during the daytime (~10:00–11:00). This heterogeneity motivated the inclusion of regional demand variables in forecasting models and suggests differing underlying consumption patterns across India's grid.

| Region        | Peak Year (Mean Demand) | Peak Month (Mean Demand) | Peak Hour (Mean Demand) | Lowest Hour (Mean Demand) | Dominant Daily Pattern |
| ------------- | ----------------------- | ------------------------ | ----------------------- | ------------------------- | ---------------------- |
| National      | 2024                    | June                     | 11:00                   | 03:00                     | Midday peak            |
| Northern      | 2023                    | August                   | 19:00                   | 03:00                     | Evening peak           |
| Southern      | 2024                    | March                    | 10:00                   | 03:00                     | Daytime peak           |
| Western       | 2024                    | February                 | 11:00                   | 03:00                     | Daytime peak           |
| Eastern       | 2023                    | July                     | 19:00                   | 06:00                     | Evening peak           |
| North-Eastern | 2023                    | August                   | 19:00                   | 04:00                     | Evening peak           |

with the following emergent seasonal trends:
| Region        | Highest Demand Month | Lowest Demand Month |
| ------------- | -------------------- | ------------------- |
| National      | June–September       | November            |
| Northern      | August–September     | November            |
| Southern      | March                | November            |
| Western       | January–February     | July                |
| Eastern       | July                 | December            |
| North-Eastern | August–September     | April / December    |


Exploratory analysis further revealed two distinct demand clusters within India:

* North, East and North-East regions exhibit evening demand peaks (~19:00)
* South and West regions exhibit daytime demand peaks (~10–11:00)

This suggests heterogeneous demand drivers across regions and motivated the inclusion of regional demand signals in forecasting models. Regional correlations were confirmed by plotting a heatmap, shown below:

<img width="765" height="677" alt="Screenshot 2026-06-02 at 3 23 02 PM" src="https://github.com/user-attachments/assets/ce602b68-23dd-4e50-a971-2250600e704d" />

---

## Forecasting Problem
We wish to predict national electricity demand 24 hours into the future.
```
target = National Hourly Demand.shift(-24)
```

### Feature Engineering: Autoregressive Features
Before constructing forecasting features, the demand series was analysed using the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)**.

The objective was to identify whether historical demand values contained predictive information and to determine appropriate lag structures for the forecasting model.

* The ACF measures the correlation between a time series and its past values at different lags. Mathematically:
$ACF(k) = Corr(y_t, y_{t-k})$ where k is the lag.
The ACF plot is shown below:

<center>
<img width="576" height="432" alt="Screenshot 2026-06-02 at 3 28 50 PM" src="https://github.com/user-attachments/assets/89e57eff-1efa-4272-824a-0045dedfb8de" />
</center>

These patterns indicate:

  1. Demand today resembles demand yesterday
  2. Demand this week resembles demand last week
  3. Daily and weekly seasonality are major drivers of electricity consumption

* The PACF measures the correlation between $y_t$ and $y_{t-k} after removing the effects of intermediate lags. The PACF plot is shown below:

  <img width="570" height="433" alt="Screenshot 2026-06-02 at 3 29 06 PM" src="https://github.com/user-attachments/assets/f5b091b7-6914-4501-9f04-1b474ae69795" />


The ACF/PACF analysis motivated the inclusion of explicit autoregressive features:
| Feature            | Motivation                           |
| ------------------ | ------------------------------------ |
| `lag_1`            | Captures short-term persistence      |
| `lag_24`           | Captures daily seasonality           |
| `lag_168`          | Captures weekly seasonality          |
| `rolling_mean_24`  | Represents recent daily demand level |
| `rolling_mean_168` | Represents weekly demand trend       |
| `rolling_std_24`   | Captures short-term volatility       |

The autocorrelation analysis revealed that electricity demand is highly persistence-driven. A strong seasonal-naive benchmark and prominent ACF spikes at 24-hour and 168-hour lags suggest that much of the forecasting signal originates from recurring daily and weekly consumption patterns.

This finding justified the use of autoregressive and rolling-window features as the foundation of the forecasting framework.

### Naive Baseline

A persistence benchmark was established by forecasting demand 24 hours ahead using the currently observed national demand:

```
y_pred = Demand(t)
```

This was evaluated against the true demand 24 hours later:

```
y_true = Demand(t+24)
```

Despite requiring no training or feature engineering, the persistence baseline achieved a MAPE of approximately 2.66%, demonstrating the highly persistent nature of short-term electricity demand.

<img width="1107" height="466" alt="Screenshot 2026-06-02 at 3 37 13 PM" src="https://github.com/user-attachments/assets/0f3c6732-4bd9-45c7-994b-4f66920dbd9d" />

The strong performance of this benchmark, together with ACF/PACF analysis, suggests that daily and weekly seasonality dominate short-horizon forecasting performance. Consequently, autoregressive lag features (`lag_1`, `lag_24`, `lag_168`) and rolling-window statistics formed the foundation of subsequent machine learning models.

