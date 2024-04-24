# Sales Prediction for Corporación Favorita

## Project Overview

This project aims to accurately forecast sales for hundreds of product families at Corporación Favorita in Ecuador. By predicting sales trends, we seek to optimize inventory management, ensuring sufficient stock to meet customer demand, enhance customer satisfaction, and streamline business operations. This project was our graduation initiative at the Miuul Bootcamp, where it was recognized as the top project.

## Business Problem

Ineffective inventory management can lead to overstocking, which increases operational costs, or understocking, which leads to missed sales opportunities and customer dissatisfaction. This project addresses these challenges by providing reliable sales forecasts for a diverse product range at Corporación Favorita.

## Dataset Description

- **train.csv**: Time series data including store number, product family, promotional status, and sales (target variable).
- **transaction.csv**: Records daily transactions per store.
- **stores.csv**: Provides metadata about each store, including city, state, type, and cluster.
- **oil.csv**: Daily oil prices, critical due to Ecuador’s oil-dependent economy.
- **holidays_events.csv**: Information on holidays and events, including transferred holidays.

## Data Preprocessing Steps

### Date Conversion and Interpolation
- Convert date-related columns across all datasets from strings to datetime objects to enable time-series manipulation.
- Interpolate missing daily oil prices and forward-fill the first day's missing value to handle gaps in the data.

### Transaction Data Cleaning
- Merge sales data with transaction counts and zero out transactions on days with zero sales to align transaction counts with sales data.
- Interpolate missing transaction data to fill in gaps, ensuring continuity in the transaction data.

### Holiday Data Adjustments
- Adjust holiday entries by removing specific holidays and modifying descriptions to standardize the data.
- Introduce flags for specific significant events (like earthquakes) to capture their potential impact on sales.

### Oil Price Features
- Calculate lagged oil prices at various intervals (15, 30, 60, 90 days) to capture the potential delayed impact of global oil price changes on local sales patterns.
- Use backward fill for missing lagged values to ensure model stability.

## Feature Engineering

- Develop sophisticated lagged sales features, rolling mean calculations, and exponentially weighted means to capture historical sales trends.
- Introduce interaction terms between promotions and transactions, and other numerical features to explore synergistic effects.
- Apply one-hot encoding to categorical variables to prepare them for machine learning modeling.

## Model Training and Evaluation

For this project, we created a customized model for each combination of store and product family. With 10 stores and 23 distinct product families, this approach resulted in the development of 230 individual models. Each model is trained to predict sales specifically tailored to the trends and seasonality of its respective store and product family.

- Utilize time series cross-validation to ensure robust model training.
- Optimize model parameters using Optuna for various models such as LightGBM, RandomForest, and XGBoost.
- Evaluate models based on RMSE, MAE, and MSE and visualize predictions to assess performance.

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, Statsmodels
- Optuna for hyperparameter optimization
- Power BI for data visualization and reporting
- Streamlit

## How to Run the Project

```bash
git clone [Repository URL]
pip install -r requirements.txt
jupyter notebook
```

## Contributors
This project is maintained by Sefa Berk Acar, Sevda Nur Çopur, Ömer Aydoğdu, and Metin Kuriş. Contributions are welcome. Please open an issue first to discuss proposed changes.
