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

### Model Development and Feature Ablation
To understand which information sources contribute most to predictive performance, a sequence of ablation experiments was performed in which feature groups were progressively added and removed.

| Model                                          | Key Features Added                                                                    | Validation MAPE |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- | --------------- |
| Persistence Baseline                           | Demand(t) → Demand(t+24)                                                              | **2.66%**       |
| XGBoost Baseline                               | National demand, temporal features, lag features, rolling statistics                  | **2.66%**       |
| XGBoost w/o National Demand                    | Temporal features, lag features, rolling statistics                                   | **2.87%**       |
| XGBoost + Regional Demand                      | National demand, regional demand, temporal features, lag features, rolling statistics | **2.59%**       |
| XGBoost + Regional Demand (No National Demand) | Regional demand, temporal features, lag features, rolling statistics                  | **2.70%**       |

**Key findings**
* Removing the national demand signal degraded performance from 2.66% to 2.87% MAPE, indicating that the current system state contains substantial information about future demand.
* Adding regional demand variables improved performance from 2.66% to 2.59% MAPE. This suggests that regional load patterns contain information not fully captured by aggregate national demand alone. The result aligns with exploratory analysis, which identified distinct demand regimes across Indian regions, and was confirmed by SHAP analysis:

<img width="789" height="729" alt="Screenshot 2026-06-02 at 3 48 20 PM" src="https://github.com/user-attachments/assets/bd19e045-be79-489e-aa77-b490724ab965" />

* When national demand was removed entirely, the model using only regional demand achieved 2.70% MAPE, substantially outperforming the model without either national or regional demand (2.87% MAPE). This indicates that regional demand signals retain much of the information contained within aggregate demand and can serve as an effective proxy for system-wide conditions.
* Machine learning and additional feature sets provide meaningful improvements, but most predictive power originates from the underlying temporal structure of the demand series.

### Weather Feature Engineering

Historical temperature data were collected using the Open-Meteo API for five geographically distributed Indian cities:
* Delhi
* Mumbai
* Chennai
* Kolkata
* Guwahati

These locations were selected to capture climatic variation across major demand regions of the Indian grid.

To capture the non-linear relationship between temperature and electricity demand, raw temperature observations were transformed into weather-derived demand indicators.

* **Cooling Degree Days (CDD)**: Cooling Degree Days ($CDD = max(T-T_{base},0)$) measure the extent to which temperatures exceed a reference comfort threshold. CDD acts as a proxy for cooling demand arising from air-conditioning usage during hot weather.
* **Heating Degree Days (HDD)**: Heating Degree Days ($HDD = max(T_{base}-T,0)$) measure the extent to which temperatures fall below the reference threshold. HDD captures additional electricity demand associated with heating requirements during colder conditions.
* **Non-Linear Temperature Effects**: Electricity demand often increases disproportionately during extreme temperatures. To capture this behaviour, a quadratic cooling term $CDD^2$ was introduced. This allows the model to represent accelerating demand growth during severe heat events, where cooling loads increase non-linearly.

HDD, CDD and CDD² allow the model to learn these asymmetric and non-linear responses directly.

Weather-derived features improved model interpretability and provided a physically meaningful representation of temperature sensitivity. While persistence and system-state variables remain the dominant drivers of forecasting performance, HDD/CDD features enabled the model to explicitly capture demand responses to temperature extremes.

### Further Modelling
To investigate the contribution of weather information, several feature representations were evaluated ranging from raw temperature observations to physically motivated demand indicators based on Heating Degree Days (HDD) and Cooling Degree Days (CDD).

| Model                                 | Weather Representation              | Validation MAPE |
| ------------------------------------- | ----------------------------------- | --------------- |
| XGBoost + Regional + Raw Weather                | City-level temperatures             | **2.75%**       |
| XGBoost + Regional + Engineered Weather         | HDD, CDD, CDD², regional aggregates | **2.85%**       |
| XGBoost + Regional + Aggregate Weather Features | India-wide HDD, CDD, CDD²           | **2.82%**       |
| XGBoost + Full Feature Diagnostics              | Extended weather feature set        | **2.83%**       |

> Note: we replace temporal signals with periodic variables here.

**Key findings**
* The strongest weather-enhanced model used raw city-level temperature observations directly and achieved a validation MAPE of 2.75%. Surprisingly, replacing these variables with engineered HDD/CDD-based features led to a small deterioration in performance. This suggests that the gradient-boosted tree model was able to learn temperature-demand relationships directly from raw temperature observations without requiring extensive feature transformation.
* Weather variables provide additional context but do not fundamentally alter predictability at a 24-hour forecasting horizon.
* Although HDD/CDD-based models did not outperform raw temperatures, they remain valuable from an energy systems perspective.

A notable outcome of this study is that increasingly sophisticated weather features did not produce commensurate improvements in forecasting accuracy. This suggests that short-term electricity demand forecasting in India is largely governed by persistence and system-state variables, with weather acting as a secondary modifier rather than a primary driver.

However, weather features remain essential for understanding demand sensitivity and enabling climate-aware scenario analysis, making them valuable despite their modest contribution to pure forecasting performance.

---

## Summary and Diagnostics
While validation performance remained strong, forecast accuracy deteriorated on the 2024 holdout period. This suggests the presence of regime shifts and evolving demand dynamics not fully represented in the training data. The result highlights a key challenge of real-world energy forecasting: models must operate in environments where consumption patterns change over time.

Model 3 (XGBoost + Regional Demand) and Model 5 (XGBoost + Regional + Raw Weather) were used to forecast energy demands for the 2024 holdout period. The results are shown below:

* Model 3 (XGBoost + Regional Demand): MAPE: 3.87%

<img width="1105" height="469" alt="Screenshot 2026-06-02 at 4 05 13 PM" src="https://github.com/user-attachments/assets/1d7b593e-24d1-4b0a-82d8-70085683be4f" />


* Model 5 (XGBoost + Regional + Raw Weather): MAPE: 3.92%

<img width="1118" height="467" alt="Screenshot 2026-06-02 at 4 05 31 PM" src="https://github.com/user-attachments/assets/865e5e81-3aba-468c-9edf-dd1fca1b1785" />

---
# Scenario Aware Modelling

Initially, the model solves a conventional supervised learning problem: $Demand_{t+24} = f(X_t)$, where $X_t$ contains current demand, regional demand, lag features, rolling statisitcs, weather variables, and calendar features. The model produces a single 24-hour-ahead forecast.

However, the model has no mechanism for evolving demand under alternative future conditions, since it only predicts one step ahead using historical observations.

To address this, a recursive simulation engine was developed by feeding model predictions back into lag and rolling-window features at each timestep. This transformed the forecasting model into a simplified demand simulator capable of generating multi-day trajectories under alternative climate scenarios, enabling stress-testing of electricity demand under sustained heatwave and coldwave conditions. Instead of producing a single forecast:
1. Predict future demand.
2. Feed the prediction back into the system.
3. Recompute lag and rolling-window features.
4. Predict the next timestep.
5. Repeat.

Because temperature is explicitly represented in the feature set of Model 5 (XGBoost + Regional + Raw Weather), weather conditions can be perturbed before simulation.

Scenarios considered included:
| Scenario        | Temperature Shift |
| --------------- | ----------------- |
| Baseline        | 0°C               |
| Mild Heatwave   | +2°C              |
| Severe Heatwave | +4°C              |
| Mild Coldwave   | -2°C              |
| Severe Coldwave | -4°C              |
> Predicted national demand is recursively fed back into autoregressive features and rolling statistics, while regional demand trajectories are held fixed at their baseline values. The simulation therefore captures feedback through national demand persistence but does not model dynamic interactions between regions.

* **Heatwave Scenario**: The energy demand forecast is shown below:

<img width="1027" height="469" alt="Screenshot 2026-06-02 at 4 16 05 PM" src="https://github.com/user-attachments/assets/35d6f700-32a5-4aa8-a392-cdc344822c3e" />

with comparisons against the baseline:

<img width="1004" height="373" alt="Screenshot 2026-06-02 at 4 16 22 PM" src="https://github.com/user-attachments/assets/2d8295ba-4bd6-4fe2-bb0a-63d2cbc44987" />

* **Coldwave Scenario**: The energy demand forecast is shown below:

<img width="1032" height="470" alt="Screenshot 2026-06-02 at 4 16 39 PM" src="https://github.com/user-attachments/assets/01759718-02a5-4ff4-9e11-eb00e6b9c03e" />

with comparisons against the baseline:

<img width="1018" height="373" alt="Screenshot 2026-06-02 at 4 16 54 PM" src="https://github.com/user-attachments/assets/699479d3-b817-439d-92d7-1cfba4cf94ed" />

---
# Limitations

* Strong Dependence on Historical Persistence: The forecasting models derive most of their predictive power from demand persistence and seasonal structure. While this produces strong short-term accuracy, it also means the models are primarily learning historical demand behaviour rather than underlying causal drivers. As a result, performance may deteriorate when demand patterns change significantly.

* Evidence of Regime Shifts: Model performance was substantially stronger on the validation period than on the 2024 holdout period. This suggests the presence of non-stationary demand dynamics and regime shifts that are not fully captured by historical training data.

* Limited Exogenous Variables: The current framework incorporates temperature as the primary external driver. However, electricity demand is also influenced by factors such as economic activity, industrial production, public holidays, policy changes, and electrification trends. These variables were not included and may explain part of the performance degradation observed in later periods.

* Simplified Weather Representation: Weather conditions were represented using a small set of city-level temperature series and derived HDD/CDD indicators. While sufficient for exploratory modelling, a more comprehensive treatment could incorporate humidity, wind speed, precipitation etc.

* Recursive Forecast Error Accumulation: The scenario simulator generates multi-day forecasts by recursively feeding model predictions back into lag and rolling-window features. This introduces error accumulation; as the forecast horizon increases, small prediction errors can propagate and amplify. For this reason, scenario results should be interpreted as exploratory stress tests rather than precise long-range forecasts.

* Simplified Regional Dynamics: The current scenario framework recursively updates only national demand features. Regional demand variables are treated as exogenous inputs and are not evolved through time. A more complete framework would jointly forecast regional and national demand, allowing temperature shocks to propagate through regional demand dynamics before aggregating to the national level.

---
# Conclusion
This project demonstrates that short-term electricity demand forecasting in India is primarily driven by persistence and system-state variables. While machine learning provides incremental improvements over strong benchmark methods, the greatest opportunities for future development lie in regime-aware forecasting, richer exogenous data integration, and scenario-based energy system analysis. By extending conventional forecasting models into a recursive simulation framework, the project illustrates how machine learning can support both operational forecasting and exploratory climate stress testing.
