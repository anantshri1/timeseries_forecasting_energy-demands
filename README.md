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


