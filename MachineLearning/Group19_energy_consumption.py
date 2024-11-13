import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

#This part is to load the dataset with the timestamp or date column
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, infer_datetime_format=True,
                           low_memory=False, na_values=['nan','?'], index_col='datetime')
#Here the relevant columns are selected 
        data = data[['Global_active_power']]
#Here the column is renamed for clarity
        data.rename(columns={'Global_active_power': 'consumption'}, inplace=True)
#Here the rows are dropped with missing values 
        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

#Here the basic information about the dataset is displayed
def display_basic_info(data):
    print("Basic Information of the Dataset:")
    print("-------------------------------")
    print("Number of rows and columns:", data.shape)
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    print("\nSummary statistics of the dataset:")
    print(data.describe())
    print("\nMissing values per column:")
    print(data.isnull().sum())
    print("-------------------------------\n")

#This part is to check for stationarity using Augmented Dickey Fuller test
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

#This part is to plot the electricity consumption data
def plot_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['consumption'], label='Daily Consumption')
    plt.xlabel('Date')
    plt.ylabel('Electricity Consumption')
    plt.title('Daily Electricity Consumption Data')
    plt.legend()
    plt.grid(True)
    plt.show()

#This part is to Plot the ACF and PACF of the scaled data
def plot_acf_pacf(scaled_consumption):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    plot_acf(scaled_consumption, lags=20, alpha=0.05, ax=ax1)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('ACF of Scaled Electricity Consumption')

    plot_pacf(scaled_consumption, lags=15, alpha=0.05, ax=ax2)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    ax2.set_title('PACF of Scaled Electricity Consumption')

    plt.tight_layout()
    plt.show()

#Here the model is trained using the Ridge Regression model with Grid Search
def train_ridge_regression(train_data, alpha_values):
    ridge_params = {"alpha": alpha_values}
    ridge_grid_search = GridSearchCV(Ridge(), ridge_params, cv=5, scoring="neg_mean_squared_error")
    ridge_grid_search.fit(train_data.drop(columns=["consumption"]), train_data["consumption"])
    return ridge_grid_search

#Here the model is evaluated
def evaluate_model(test_data, forecast, model_name):
    test_data = test_data.copy()  
    test_data['forecast'] = forecast
    
    residuals = test_data['consumption'] - test_data['forecast']
    mae = mean_absolute_error(test_data['consumption'], test_data['forecast'])
    mse = mean_squared_error(test_data['consumption'], test_data['forecast'])
    rmse = np.sqrt(mse)
    print(f"{model_name} Metrics:")
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data['consumption'], label='Actual Consumption', color='blue')
    plt.plot(test_data.index, test_data['forecast'], label=f'Predicted Consumption ({model_name})', color='red')
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.title(f'Actual vs. Predicted Electricity Consumption ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, residuals, label=f'{model_name} Residuals', color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.title(f'Residuals of {model_name} Model')
    plt.legend()
    plt.grid(True)
    plt.show()

#This is the main function
def main():
    file_path = "household_power_consumption.txt"  
    data = load_data(file_path)
    if data is None:
        return
    
#This part is to resample to daily consumption
    daily_consumption = data.resample('D').sum()
    display_basic_info(daily_consumption)

#This part is to ensure the dataset is sorted chronologically
    daily_consumption.sort_index(inplace=True)

#Here feature scaling is applied to the consumption data
    scaler = StandardScaler()
    daily_consumption['scaled_consumption'] = scaler.fit_transform(daily_consumption[['consumption']])

#This part is to check for stationarity using ADF test
    print("Stationarity Check:")
    check_stationarity(daily_consumption['scaled_consumption'])

#Here the data is plotted
    plot_data(daily_consumption)
    
#Here the ACF and PACF are plotted
    plot_acf_pacf(daily_consumption['scaled_consumption'])
    
#This part calculates and prints the PACF and ACF values
    pacf_values = pacf(daily_consumption['scaled_consumption'])
    acf_values = acf(daily_consumption['scaled_consumption'])
    print("Partial Autocorrelation Function (PACF) of Scaled Consumption:")
    print(pacf_values)
    print("Autocorrelation Function (ACF) of Scaled Consumption:")
    print(acf_values)

#This part splits the data into training and testing sets based on days
    train_size = int(len(daily_consumption) * 0.8)
    train_data, test_data = daily_consumption.iloc[:train_size], daily_consumption.iloc[train_size:]

#This part adds previous day consumption as a feature for Ridge Regression
    train_data['prev_consumption'] = train_data['consumption'].shift(1).fillna(0)
    test_data['prev_consumption'] = test_data['consumption'].shift(1).fillna(0)

#This part trains and evaluates the Ridge Regression model
    alpha_values = [0.1, 0.5, 1.0]  # Adjust the range as needed
    ridge_grid_search = train_ridge_regression(train_data, alpha_values)

    print("Best parameters for Ridge Regression:", ridge_grid_search.best_params_)
    ridge_model = ridge_grid_search.best_estimator_

#Here the grid search results are saved to CSV
    results_df = pd.DataFrame(ridge_grid_search.cv_results_)
    columns_of_interest = ['param_alpha', 'mean_test_score', 'std_test_score', 'rank_test_score']
    filtered_results = results_df[columns_of_interest]
    filtered_results.to_csv('ridge_grid_search_results.csv', index=False)
    print("The results of the grid search have been saved to 'ridge_grid_search_results.csv'.")

#This parts makes predictions for future electricity consumption using Ridge Regression
    ridge_forecast = ridge_model.predict(test_data.drop(columns=['consumption']))
    evaluate_model(test_data, ridge_forecast, "Ridge Regression")

#Here the SARIMA model parameters are defined
    order = (1, 0, 1)
    seasonal_order = (1, 1, 1, 12)

    sarima_model = SARIMAX(train_data['scaled_consumption'], order=order, seasonal_order=seasonal_order)
    sarima_model_result = sarima_model.fit()
    print(sarima_model_result.summary())
    
    forecast = sarima_model_result.forecast(steps=len(test_data))
    forecast = scaler.inverse_transform(forecast.values.reshape(-1, 1)).flatten()
    evaluate_model(test_data, forecast, "SARIMA")

    rolling_predictions = test_data.copy()
    for i, (index, _) in enumerate(test_data.iterrows()):
        train_data = daily_consumption.iloc[:train_size + i]
        sarima_model = SARIMAX(train_data['scaled_consumption'], order=order, seasonal_order=seasonal_order)
        sarima_model_result = sarima_model.fit()
        pred = sarima_model_result.forecast()
        pred = scaler.inverse_transform(pred.values.reshape(-1, 1)).flatten()  # Reshape the predicted values
        rolling_predictions.at[index, 'forecast'] = pred[0]

    rolling_residuals = test_data['consumption'] - rolling_predictions['forecast']
    sarima_mae = mean_absolute_error(test_data['consumption'], rolling_predictions['forecast'])
    sarima_mse = mean_squared_error(test_data['consumption'], rolling_predictions['forecast'])
    sarima_rmse = np.sqrt(sarima_mse)
    print("SARIMA Metrics (Rolling Origin Forecast):")
    print(f"MAE: {sarima_mae}, MSE: {sarima_mse}, RMSE: {sarima_rmse}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data['consumption'], label='Historical Consumption')
    plt.plot(test_data.index, test_data['consumption'], label='Actual Consumption')
    plt.plot(test_data.index, rolling_predictions['forecast'], label='Predicted Consumption (SARIMA)', color='red')
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.title('Actual vs. Predicted Electricity Consumption (SARIMA: Rolling Origin Forecast)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, rolling_residuals, label='Rolling Forecast Residuals', color='purple')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.title('Residuals of Rolling Forecast Model (SARIMA)')
    plt.legend()
    plt.grid(True)
    plt.show()

    forecast_horizon = 7
    last_date = daily_consumption.index[-1]
    extended_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

    extended_data = pd.DataFrame(index=extended_index)

    rolling_predictions_extended = extended_data.copy()
    sarima_model_extended = SARIMAX(daily_consumption['scaled_consumption'], order=order, seasonal_order=seasonal_order)
    sarima_model_result_extended = sarima_model_extended.fit()
    forecast_extended = sarima_model_result_extended.forecast(steps=forecast_horizon)
    forecast_extended = scaler.inverse_transform(forecast_extended.values.reshape(-1, 1)).flatten()
    rolling_predictions_extended['forecast'] = forecast_extended

    plt.figure(figsize=(10, 6))
    plt.plot(daily_consumption.index, daily_consumption['consumption'], label='Historical Consumption')
    plt.plot(rolling_predictions_extended.index, rolling_predictions_extended['forecast'], label='Forecasted Consumption', color='red')
    plt.xlabel('Date')
    plt.ylabel('Consumption')
    plt.title('Forecasted Electricity Consumption for Next Weeks (SARIMA)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
