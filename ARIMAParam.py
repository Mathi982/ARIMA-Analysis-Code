import pandas as pd
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


# Function to automatically determine ARIMA parameters and plot ACF and PACF
def determine_arima_parameters(file_path):
    data = pd.read_csv(file_path)
    data['Created Date'] = pd.to_datetime(data['Created Date'], format='%d/%m/%Y')
    daily_registrations = data.groupby('Created Date').size()

    # Plotting ACF and PACF for manual inspection
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(daily_registrations, ax=plt.gca(), title="ACF for " + file_path.split('/')[-1])
    plt.subplot(122)
    plot_pacf(daily_registrations, ax=plt.gca(), title="PACF for " + file_path.split('/')[-1])
    plt.show()

    # Using auto_arima to determine the best parameters
    auto_model = auto_arima(daily_registrations, seasonal=False, stepwise=True, suppress_warnings=True,
                            error_action="ignore", trace=True)

    return auto_model.order


# File paths for each dataset
file_paths = ['/Users/Mathi/Desktop/D19.csv', '/Users/Mathi/Desktop/D21.csv', '/Users/Mathi/Desktop/GP21.csv',
              '/Users/Mathi/Desktop/MSE21.csv', '/Users/Mathi/Desktop/NP21.csv', '/Users/Mathi/Desktop/SRM22.csv',
              '/Users/Mathi/Desktop/SRM23.csv']

# Loading each dataset, plotting ACF and PACF, and determining ARIMA parameters
for path in file_paths:
    order = determine_arima_parameters(path)
    print(f"Best ARIMA parameters for {path.split('/')[-1]}: {order}")
