# Description: Short example for Nixtla Suite for Time Series Forecasting with Python.


import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_io import read_csv
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, HistoricAverage, HoltWinters, SeasonalNaive
from statsforecast.models import CrostonClassic as Croston
from statsforecast.models import DynamicOptimizedTheta as DOT
from utilsforecast.losses import mape

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Set environment variable
os.environ["NIXTLA_ID_AS_COL"] = "1"

# Load and preprocess the data
df = read_csv("ercot_load_data.csv")
df["ds"] = pd.to_datetime(df["date"])  # Ensure 'date' is in datetime format
df["y"] = pd.to_numeric(df["values"], errors="coerce")  # Convert 'values' to numeric
df = df.sort_values("ds")  # Sort by date
df = df.dropna(subset=["ds", "y"])  # Drop rows with missing values

# Resample the data to hourly frequency
df = df.set_index("ds").resample("h")["y"].mean().reset_index()
df["unique_id"] = "series1"  # Assign a unique ID for StatsForecast compatibility

# Split the data into training and hold-out sets
hold_out_hours = 24
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

# Initialize and fit the StatsForecast model
models = [AutoARIMA(season_length=24)]  # Adjust seasonality to daily
sf = StatsForecast(models=models, freq="h", n_jobs=-1)
sf.fit(train)

# Generate forecasts for the hold-out period
horizon = len(hold_out)
forecasts = sf.predict(h=horizon)

# Add timestamps to the forecast results
forecasts["ds"] = hold_out["ds"].values

# Visualize the results
plt.figure(figsize=(12, 6))

# Plot historical data
plt.plot(df["ds"], df["y"], label="Historical Data", color="blue")

# Highlight hold-out data in green
plt.plot(hold_out["ds"], hold_out["y"], label="Hold-Out Data", color="green")

# Plot forecasted data in red
plt.plot(forecasts["ds"], forecasts["AutoARIMA"], label="Forecast", color="red")

# Add labels, title, and legend
plt.title("Time Series Forecast with AutoARIMA")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("AutoARIMA_Forecast.png")
plt.show()

# Calculate and print forecast metrics
actual_values = hold_out["y"].values
forecast_values = forecasts["AutoARIMA"].values

mse_value = np.mean((actual_values - forecast_values) ** 2)
rmse_value = np.sqrt(mse_value)
mae_value = np.mean(np.abs(actual_values - forecast_values))

logger.info("\nForecast Metrics:")
logger.info(f"Mean Squared Error (MSE): {mse_value:.2f}")
logger.info(f"Root Mean Squared Error (RMSE): {rmse_value:.2f}")
logger.info(f"Mean Absolute Error (MAE): {mae_value:.2f}")

# Forecast Metrics:
# Mean Squared Error (MSE): 142.56
# Root Mean Squared Error (RMSE): 11.94
# Mean Absolute Error (MAE): 11.00


# Create a list of models and instantiation parameters
models = [
    HoltWinters(),
    Croston(),
    SeasonalNaive(season_length=24),
    HistoricAverage(),
    DOT(season_length=24),
]
# Instantiate StatsForecast class as sf
sf = StatsForecast(
    models=models,
    freq="h",
    fallback_model=SeasonalNaive(season_length=7),
    n_jobs=-1,
)

forecasts_df = sf.forecast(df=train, h=48, level=[90])
forecasts_df.head()

sf.plot(df, forecasts_df)

cv_df = sf.cross_validation(df=df, h=24, step_size=24, n_windows=2)


def evaluate_cv(df, metric):
    models = df.columns.drop(["unique_id", "ds", "y", "cutoff"]).tolist()
    evals = metric(df, models=models)
    evals["best_model"] = evals[models].idxmin(axis=1)
    return evals



def main():
    evaluation_df = evaluate_cv(cv_df, mape)
    evaluation_df.head()

    sf.plot(df, forecasts_df, models=["DynamicOptimizedTheta"], level=[90])

    # Forecast Metrics:
    # Mean Squared Error (MSE): 598.98
    # Root Mean Squared Error (RMSE): 24.47
    # Mean Absolute Error (MAE): 19.99


if __name__ == "__main__":
    main()
