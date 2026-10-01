# Farmora ML — Agricultural Price Prediction

## Overview

Farmora ML is the machine learning component of the Farmora project, focused on agricultural market-price prediction and price-trend intelligence.

The ML system analyzes historical agricultural market data and predicts future modal prices for selected crops.

### Crops Covered

- Cotton
- Rice
- Sugarcane
- Wheat
- Onion
- Tomato

---

## ML Objective

The primary objective is:

> **Predict future agricultural market prices using historical market-price data and machine learning.**

The ML pipeline follows:

```text
Historical Market Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Target Creation
        ↓
Time-Series Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Selection
        ↓
Future Price Prediction
        ↓
Price Trend Analysis
```

---

## ML Responsibility

The ML component is responsible for:

- Processing historical agricultural price data
- Cleaning and preparing market-price datasets
- Creating time-series features
- Creating prediction targets
- Training machine learning models
- Comparing model performance
- Selecting suitable models for different crops
- Predicting future agricultural prices
- Estimating price movement and trends
- Saving trained model artifacts
- Supporting future price-intelligence features

The ML model focuses on **price prediction**.

It does not directly determine farmer actions such as:

- SELL NOW
- WAIT
- PARTIAL SELL

Those decisions require additional information beyond the predicted price.

---

# Data Source

The primary data source used for agricultural market-price prediction is:

**AGMARKNET / Government Open Data Platform (data.gov.in)**

The dataset contains historical daily market prices for agricultural commodities.

### Important Dataset Fields

```text
State
District
Market
Commodity
Variety
Grade
Date
Min Price
Max Price
Modal Price
```

The **Modal Price** is used as the primary prediction target because it represents the prevailing market price and is suitable for farmer-facing price prediction.

---

# Dataset Scope

The ML system focuses on the following agricultural commodities:

| Crop | Prediction Horizon |
|------|---------------------|
| Cotton | 14 days |
| Rice | 14 days |
| Sugarcane | 2 days |
| Wheat | 14 days |
| Onion | 5 days |
| Tomato | 3 days |

The prediction horizon can vary between crops because agricultural commodities have different price behaviors, volatility, and market characteristics.

---

# Prediction Target

The primary prediction target is:

```text
Future Modal Price
```

For a given crop and market, the model learns the relationship between historical market information and a future modal price.

Example:

```text
Historical Prices
      ↓
₹7,200
₹7,250
₹7,180
₹7,320
₹7,400
      ↓
ML Model
      ↓
Future Predicted Price
      ↓
₹7,550
```

The predicted price is an estimate and should not be treated as a guaranteed market price.

---

# ML Dataset Pipeline

The complete dataset preparation pipeline is:

```text
Raw AGMARKNET Data
        ↓
Load Dataset
        ↓
Column Standardization
        ↓
Data Type Conversion
        ↓
Duplicate Removal
        ↓
Missing Value Handling
        ↓
Outlier Detection
        ↓
Sorting by Date
        ↓
Feature Engineering
        ↓
Target Creation
        ↓
Model-Ready Dataset
```

---

# Data Preprocessing

## 1. Column Standardization

Dataset columns are standardized so that they can be consistently processed by the ML pipeline.

Important columns include:

```text
Date
State
District
Market
Commodity
Variety
Grade
Min_Price
Max_Price
Modal_Price
```

---

## 2. Date Processing

The date column is converted into a proper datetime format.

The dataset is then sorted chronologically.

```text
Date
 ↓
Datetime Conversion
 ↓
Chronological Sorting
```

Date information can also be used to derive time-related features.

---

## 3. Duplicate Removal

Duplicate market-price records are identified and removed where necessary.

This prevents repeated records from affecting model training.

---

## 4. Missing Values

Missing values are identified during preprocessing.

Depending on the feature and the amount of missing data, suitable handling methods can be applied.

For time-series price data, short gaps can be handled using techniques such as:

```text
Forward Filling
Interpolation
Rolling Statistics
```

If a market or commodity contains insufficient historical information, the data may be excluded from model training.

---

# Outlier Handling

Agricultural prices can sometimes contain unusually large price movements.

The ML pipeline checks for implausible price changes and extreme values.

For example:

```text
Previous Price = ₹2,000
Current Price  = ₹7,000
```

Such a large movement should be examined before being treated as normal training data.

Outlier handling is important because extreme incorrect records can negatively affect model performance.

---

# Feature Engineering

Feature engineering converts raw market-price data into meaningful numerical and categorical inputs for machine learning.

The project uses historical price behavior and market information to create features.

---

## Historical Price Features

Lag features represent previous market prices.

Examples:

```text
Lag_1
Lag_2
Lag_3
Lag_7
Lag_14
Lag_30
```

For example:

```text
Lag_1  → Previous day's price
Lag_7  → Price from 7 days earlier
Lag_14 → Price from 14 days earlier
```

These features help the model understand historical price behavior.

---

# Rolling Features

Rolling statistics are calculated over historical price windows.

Examples include:

```text
7-Day Moving Average
14-Day Moving Average
30-Day Moving Average
```

Rolling features help identify short-term and medium-term price patterns.

Example:

```text
Previous Prices
      ↓
7-Day Rolling Mean
      ↓
Smoothed Price Trend
```

Additional rolling statistics can include:

```text
Rolling Mean
Rolling Standard Deviation
```

---

# Price Change Features

Price-change features help capture market momentum.

Examples:

```text
1-Day Price Change
7-Day Price Change
Price Change Percentage
```

A percentage change can be calculated as:

```text
Price Change % =
((Current Price - Previous Price) / Previous Price) × 100
```

---

# Time-Based Features

The date can be transformed into additional features such as:

```text
Year
Month
Day
Day of Week
Week of Year
```

These features can help the model identify seasonal and calendar-based price patterns.

---

# Market and Commodity Features

The dataset contains categorical information such as:

```text
Commodity
Market
District
State
Variety
Grade
```

These features provide information about where and for which commodity the price was recorded.

Categorical variables can be converted into numerical representations before model training.

---

# Model Training

The project evaluates multiple machine learning algorithms instead of relying on a single model.

The main models are:

1. Linear Regression
2. Random Forest
3. XGBoost

The models are trained and evaluated using historical market-price data.

---

# Machine Learning Models

## 1. Linear Regression

Linear Regression is used as a baseline regression model.

It attempts to learn a relationship between input features and the target price.

Conceptually:

```text
Input Features
      ↓
Linear Relationship
      ↓
Predicted Price
```

### Advantages

- Simple
- Fast to train
- Easy to interpret
- Useful as a baseline model

Linear Regression is especially useful for determining whether more complex models provide meaningful improvement.

---

# 2. Random Forest

Random Forest is an ensemble machine learning algorithm based on multiple decision trees.

The general process is:

```text
Input Features
      ↓
Multiple Decision Trees
      ↓
Individual Predictions
      ↓
Combined Prediction
      ↓
Final Price
```

Random Forest can capture nonlinear relationships between market features and prices.

### Advantages

- Handles nonlinear relationships
- Works well with tabular datasets
- Robust to noisy features
- Can model feature interactions

---

# 3. XGBoost

XGBoost is a gradient-boosting algorithm designed for high-performance tabular machine learning.

The general process is:

```text
Input Features
      ↓
Decision Tree
      ↓
Prediction Error
      ↓
Next Tree
      ↓
Prediction Improvement
      ↓
Final Prediction
```

XGBoost builds trees sequentially, with later trees attempting to improve the errors made by previous trees.

### Advantages

- Strong performance on tabular data
- Captures nonlinear relationships
- Handles feature interactions
- Suitable for regression problems

---

# Model Comparison

The three models are evaluated using the same evaluation methodology.

```text
Linear Regression
        │
        ├── MAE
        ├── RMSE
        └── R²
        
Random Forest
        │
        ├── MAE
        ├── RMSE
        └── R²
        
XGBoost
        │
        ├── MAE
        ├── RMSE
        └── R²
```

The model is selected based on validation performance for the corresponding crop.

---

# Time-Series Validation

Agricultural price data is time-dependent.

Therefore, a random train-test split is avoided for the primary time-series evaluation.

Instead, the data is divided chronologically.

Example:

```text
Older Data
    ↓
Training Data
    ↓
Validation / Test Data
    ↓
Newer Data
```

A typical structure is:

```text
80% → Training
20% → Testing
```

The important principle is:

> **The model should be trained on past data and evaluated on future data.**

This prevents future information from leaking into the training process.

---

# Why Random Train-Test Split Is Avoided

Consider a dataset containing:

```text
2022
2023
2024
2025
2026
```

A random split could place:

```text
2026 data → Training
2023 data → Testing
```

This does not accurately represent real-world forecasting.

Instead:

```text
2022 ───────── 2025
       Training

2026
Testing
```

This better represents how the model will be used to predict future prices.

---

# Evaluation Metrics

The ML models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted prices.

Formula:

```text
MAE = Average(|Actual Price - Predicted Price|)
```

Example:

```text
Actual Price     = ₹5,000
Predicted Price  = ₹4,800

Error = ₹200
```

Lower MAE indicates smaller average prediction errors.

---

# Root Mean Squared Error — RMSE

RMSE measures prediction error while giving greater importance to larger errors.

Formula:

```text
RMSE = √(Average((Actual - Predicted)²))
```

RMSE is useful for identifying whether the model produces large prediction errors.

Lower RMSE indicates better prediction performance.

---

# R² Score

R² indicates how well the model explains variation in the target variable.

General interpretation:

```text
R² close to 1
    ↓
Strong explanatory performance

R² close to 0
    ↓
Limited explanatory performance

R² below 0
    ↓
Model performs poorly relative to a simple baseline
```

R² should be interpreted together with MAE and RMSE rather than independently.

---

# Model Performance

The trained crop-specific models have been evaluated using MAE, RMSE, and R².

| Crop | Selected Model | MAE | RMSE | R² |
|------|----------------|-----|------|----|
| Cotton | Linear Regression | 365.8751 | 502.7367 | 0.7447 |
| Rice | Linear Regression | 542.95 | 995.11 | 0.5723 |
| Sugarcane | Random Forest | 116.30 | 347.60 | -0.1900 |
| Wheat | Random Forest | 86.41 | 121.68 | 0.8801 |
| Onion | Linear Regression | 251.40 | 380.56 | 0.8467 |
| Tomato | Linear Regression | 276.14 | 421.03 | 0.7299 |

The metrics represent model evaluation results for the corresponding crop-specific prediction tasks.

---

# Crop-Specific Model Selection

The project uses crop-specific model selection rather than forcing one algorithm to work identically for every crop.

Current selected models:

```text
Cotton
    ↓
Linear Regression

Rice
    ↓
Linear Regression

Sugarcane
    ↓
Random Forest

Wheat
    ↓
Random Forest

Onion
    ↓
Linear Regression

Tomato
    ↓
Linear Regression
```

This approach allows the model choice to depend on the observed performance of each crop's prediction task.

---

# Price Prediction Pipeline

The prediction process follows:

```text
Historical Market Data
        ↓
Preprocessing
        ↓
Feature Engineering
        ↓
Load Crop-Specific Model
        ↓
Generate Required Features
        ↓
Model Prediction
        ↓
Future Modal Price
```

Example:

```text
Crop: Cotton
Market: Selected Market
Current Historical Data
        ↓
Feature Generation
        ↓
Cotton Model
        ↓
Predicted Future Price
```

---

# Price Trend Analysis

The predicted price can be compared with the current market price to identify the expected direction of movement.

The percentage change is calculated as:

```text
Change Percentage =
((Predicted Price - Current Price) / Current Price) × 100
```

The resulting value can be interpreted as:

```text
Positive Change
      ↓
Expected Price Increase

Negative Change
      ↓
Expected Price Decrease

Small Change
      ↓
Expected Stable Movement
```

A commonly used threshold in the project is:

```text
Change > +2%
    → RISING

Change < -2%
    → FALLING

-2% to +2%
    → STABLE
```

This represents price movement based on the model prediction and current price.

---

# Prediction Example

Example:

```text
Current Price      = ₹6,000
Predicted Price    = ₹6,300
```

Calculation:

```text
Change % =
((6300 - 6000) / 6000) × 100

= 5%
```

Therefore:

```text
Expected Trend = RISING
```

Another example:

```text
Current Price      = ₹6,000
Predicted Price    = ₹5,700
```

```text
Change % =
((5700 - 6000) / 6000) × 100

= -5%
```

Therefore:

```text
Expected Trend = FALLING
```

---

# Confidence and Prediction Uncertainty

Machine learning predictions are estimates.

Agricultural markets can be affected by factors such as:

```text
Weather
Supply
Demand
Seasonality
Market Arrivals
Government Policies
Transportation
Local Market Conditions
```

Therefore:

> **A predicted price should be interpreted as an estimate rather than a guaranteed future price.**

Confidence or uncertainty information can be used to communicate the reliability of predictions.

---

# Explainability

The ML component is designed to remain understandable and explainable.

Important factors considered by the models include:

```text
Historical Prices
Lag Features
Moving Averages
Price Changes
Commodity
Market
District
Seasonal Information
```

The goal is to understand which historical and market characteristics contribute to predicted price movement.

---

# Model Artifacts

Trained models can be serialized and stored for later prediction.

The project uses model serialization techniques such as `joblib`.

Example model artifacts:

```text
cotton_price_model.joblib
rice_price_model.joblib
sugarcane_price_model.joblib
wheat_price_model.joblib
onion_price_model.joblib
tomato_price_model.joblib
```

The exact filenames may vary depending on the training pipeline.

---

# ML Project Structure

A typical ML project structure is:

```text
ml/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── models/
│   ├── cotton_price_model.joblib
│   ├── rice_price_model.joblib
│   ├── sugarcane_price_model.joblib
│   ├── wheat_price_model.joblib
│   ├── onion_price_model.joblib
│   └── tomato_price_model.joblib
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   └── model_training.ipynb
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── results/
│   ├── metrics/
│   └── predictions/
│
└── README.md
```

The actual project structure may differ depending on the implementation.

---

# Technology Stack

The ML component uses Python-based data science and machine learning technologies.

### Programming Language

```text
Python
```

### Data Processing

```text
Pandas
NumPy
```

### Machine Learning

```text
Scikit-learn
Random Forest
Linear Regression
XGBoost
```

### Model Serialization

```text
Joblib
```

### Data Source

```text
AGMARKNET
Government Open Data Platform
data.gov.in
```

---

# ML Workflow

The complete ML workflow is:

```text
┌──────────────────────────────┐
│      AGMARKNET Dataset       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      Data Preprocessing      │
│  Cleaning + Missing Values   │
│      + Outlier Checks        │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     Feature Engineering      │
│ Lag + Rolling + Time +       │
│ Market-Based Features        │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       Target Creation        │
│     Future Modal Price       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     Chronological Split      │
│     Train → Future Test      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       Model Training         │
│                              │
│ Linear Regression            │
│ Random Forest                │
│ XGBoost                      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      Model Evaluation        │
│                              │
│ MAE + RMSE + R²              │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     Model Selection          │
│     Crop-Specific Model      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     Future Price Prediction  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      Price Trend Analysis    │
└──────────────────────────────┘
```

---

# Why Multiple ML Models?

Agricultural commodities do not necessarily follow the same price behavior.

For example:

```text
Cotton
   ↓
Linear relationship may perform well

Wheat
   ↓
Nonlinear relationships may be better captured

Sugarcane
   ↓
Random Forest evaluated against alternatives
```

Therefore, multiple algorithms are trained and compared.

The final model is selected based on validation performance rather than assuming that the most complex algorithm will always be the best.

---

# Why Not Use Deep Learning Initially?

Models such as:

```text
LSTM
GRU
Transformers
```

can be useful for complex sequential forecasting.

However, the initial Farmora ML implementation focuses on classical machine learning because:

- The available data is primarily structured/tabular market data.
- Classical models are faster to train.
- They are easier to evaluate and explain.
- They are suitable for an initial prototype.
- Model behavior can be compared more easily.
- The project prioritizes practical and explainable price prediction.

Deep learning can be considered as a future enhancement if larger and richer time-series datasets become available.

---

# Baseline Comparison

Before relying on complex models, a simple baseline can be used for comparison.

Possible baselines include:

```text
Previous Day Price
Moving Average
Historical Average
```

Example:

```text
Today's Price
     ↓
Use as Tomorrow's Prediction
```

A trained ML model should ideally provide useful improvement over such simple forecasting approaches.

---

# Future ML Enhancements

Future versions of Farmora ML can include:

### Advanced Price Forecasting

```text
Longer Prediction Horizons
Multi-Step Forecasting
Probabilistic Forecasting
```

### Additional Data Sources

```text
Weather
Rainfall
Market Arrivals
Production Data
Demand Indicators
Government Price Information
```

### Advanced Models

```text
XGBoost Optimization
LightGBM
LSTM
GRU
Transformer-Based Time-Series Models
```

### Demand Forecasting

Future versions can estimate:

```text
Market Demand
Demand Trends
Expected Supply-Demand Changes
```

### Market-Level Forecasting

Prediction can be extended to:

```text
Commodity
        ↓
State
        ↓
District
        ↓
Market
```

This can help provide more localized price predictions.

---

# Limitations

The ML predictions have several limitations.

## Market Volatility

Agricultural prices can change rapidly due to external factors.

## Weather

Unexpected weather events can significantly affect supply and prices.

## Data Quality

Missing, incorrect, or delayed market records can affect prediction quality.

## Market Differences

Different markets can have different price behaviors.

## External Events

Government policies, transportation issues, sudden demand changes, and supply disruptions may not be fully represented in historical price features.

## Prediction Uncertainty

Machine learning predictions are estimates and cannot guarantee future market prices.

---

# ML Design Principles

The Farmora ML component follows these principles:

### 1. Data-Driven

Predictions are based on historical market data.

### 2. Time-Aware

Training and testing respect chronological order.

### 3. Crop-Specific

Different crops can use different trained models.

### 4. Explainable

The pipeline uses understandable features and standard regression metrics.

### 5. Evaluated

Models are compared using objective validation metrics.

### 6. Extensible

The architecture allows additional features and models to be added later.

### 7. Practical

The initial implementation prioritizes reliable tabular ML rather than unnecessary complexity.

---

# Complete ML Flow

The Farmora ML process can be summarized as:

```text
                    ┌──────────────────────┐
                    │  Historical Market   │
                    │       Data           │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Data Preprocessing   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Future Price Target  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Time-Series Split    │
                    └──────────┬───────────┘
                               ↓
              ┌────────────────────────────────┐
              │        Model Training          │
              │                                │
              │ Linear Regression              │
              │ Random Forest                  │
              │ XGBoost                        │
              └────────────────┬───────────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Model Evaluation     │
                    │ MAE / RMSE / R²      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Crop-Specific Model  │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Future Price        │
                    │ Prediction           │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Price Trend          │
                    │ Analysis             │
                    └──────────────────────┘
```

---

# Current ML Status

The ML component covers:

- [x] Agricultural market-price dataset preparation
- [x] Data preprocessing
- [x] Feature engineering
- [x] Historical price features
- [x] Time-series based validation
- [x] Linear Regression
- [x] Random Forest
- [x] XGBoost evaluation
- [x] MAE evaluation
- [x] RMSE evaluation
- [x] R² evaluation
- [x] Crop-specific model selection
- [x] Future price prediction
- [x] Price trend calculation
- [x] Model artifact generation

---

# Key Takeaway

Farmora ML transforms historical agricultural market data into future price estimates through a structured machine learning pipeline.

```text
Historical Data
      ↓
Clean & Prepare
      ↓
Engineer Features
      ↓
Train Multiple Models
      ↓
Evaluate Performance
      ↓
Select Crop-Specific Model
      ↓
Predict Future Price
      ↓
Analyze Expected Price Trend
```

> **Farmora ML is designed to turn historical agricultural market data into practical, data-driven price intelligence for farmers.**