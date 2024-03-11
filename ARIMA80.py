import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")


# Function to conduct ADF test, fit ARIMA model, and make forecasts for the 80% threshold
def forecast_with_arima_adjusted(file_path, params):
    # Load data
    data = pd.read_csv(file_path)
    data['Created Date'] = pd.to_datetime(data['Created Date'], format='%d/%m/%Y')

    # Group by 'Created Date' to get daily registrations
    daily_registrations = data.groupby('Created Date').size()

    # Ensure that the time series is continuous with no missing dates
    all_dates = pd.date_range(start=daily_registrations.index.min(), end=daily_registrations.index.max(), freq='D')
    daily_registrations = daily_registrations.reindex(all_dates, fill_value=0)

    # Calculate the 80% threshold index for the time series
    threshold_index = int(len(daily_registrations) * 0.8)

    # Split the data into training and testing sets at the 80% mark
    train = daily_registrations[:threshold_index]
    test = daily_registrations[threshold_index:]

    # Fit ARIMA model to the training set
    model = ARIMA(train, order=params)
    model_fit = model.fit()

    # Forecast for the length of the test set
    forecast = model_fit.forecast(steps=len(test))

    # ADF Test
    adf_test_result = adfuller(train.dropna())
    print(f"ADF Statistic: {adf_test_result[0]}")
    print(f"p-value: {adf_test_result[1]}")

    # Calculate error metrics based on the test set
    rmse = sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    mape = np.mean(np.abs(forecast - test) / np.abs(test)) * 100

    # AIC and BIC
    aic = model_fit.aic
    bic = model_fit.bic

    # Print the statistics
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    print(f"AIC: {aic}")
    print(f"BIC: {bic}")

    # Plotting the historical data and forecasts
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data', color='blue')
    plt.plot(test.index, test, label='Actual Test Data', color='green')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title(f'ARIMA Model Forecast from 80% Threshold for {file_path.split("/")[-1]}')
    plt.xlabel('Date')
    plt.ylabel('Number of Registrations')
    plt.legend()
    plt.grid(True)
    plt.show()

    return forecast


# File paths in the current environment
file_paths = ['/Users/Mathi/Desktop/D19.csv', '/Users/Mathi/Desktop/D21.csv', '/Users/Mathi/Desktop/GP21.csv', '/Users/Mathi/Desktop/MSE21.csv',
              '/Users/Mathi/Desktop/NP21.csv', '/Users/Mathi/Desktop/SRM22.csv', '/Users/Mathi/Desktop/SRM23.csv']


# Best ARIMA parameters for each dataset
arima_params = {
    'D19': (1, 0, 0),
    'D21': (0, 1, 1),
    'GP21': (1, 0, 0),
    'MSE21': (1, 0, 2),
    'NP21': (0, 0, 0),
    'SRM22': (0, 0, 0),
    'SRM23': (2, 0, 2)
}

# Run the forecasting for each dataset
forecast_results = {}
for file_path in file_paths:
    dataset_name = file_path.split('/')[-1].split('.')[0]
    print(f"Forecasting for {dataset_name}")
    forecast_results[dataset_name] = forecast_with_arima_adjusted(file_path, arima_params[dataset_name])
