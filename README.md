# ARIMA-Analysis-Code

This Python code performs time series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model on multiple datasets for a client that wanted to see if they could forecast the number of final registrations for a conference based on the way the registrations are being made prior to the stat date of the conference.

Importing Libraries: 
* The code imports necessary libraries such as pandas for data manipulation, numpy for numerical computations, statsmodels for time series analysis, matplotlib for plotting, and sklearn for error metrics computation.

forecast_with_arima_adjusted Function:

* This function takes two arguments: file_path (path to the dataset file) and params (dictionary containing ARIMA model parameters for different datasets).
* It loads the dataset from the specified file path, converts the 'Created Date' column to datetime format, and groups the data by date to get daily registrations.
* Ensures that the time series is continuous with no missing dates by reindexing.
* Splits the data into training and testing sets using an 80% threshold.
* Fits an ARIMA model to the training set using parameters specified in the params dictionary.
* Forecasts for the length of the test set using the fitted model.
* Performs Augmented Dickey-Fuller (ADF) test for stationarity on the training data.
* Calculates error metrics such as RMSE, MAE, and MAPE.
* Computes AIC and BIC values.
* Plots historical data, actual test data, and forecasts.
* Returns the forecast.

file_paths and arima_params:
* file_paths contains the file paths for the datasets.
* arima_params is a dictionary containing the best ARIMA model parameters for each dataset.

Forecasting for Each Dataset:
* It iterates through each dataset file path.
* Extracts the dataset name from the file path.
* Prints the dataset name.
* Calls the forecast_with_arima_adjusted function with the dataset file path and corresponding ARIMA parameters.
* Stores the forecast results in a dictionary.

Output:
* The code prints forecast results, including ADF statistic, p-value, error metrics (RMSE, MAE, MAPE), AIC, and BIC for each dataset.
* It plots the historical data, actual test data, and forecasts for each dataset.
