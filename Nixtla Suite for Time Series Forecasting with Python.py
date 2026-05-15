"""Generated from Jupyter notebook: Nixtla Suite for Time Series Forecasting with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

# !pip install statsforecast  # Jupyter-only
# !pip install neuralforecast  # Jupyter-only
# !pip install mlforecast  # Jupyter-only
# !pip install hierarchicalforecast  # Jupyter-only
# !pip install datasetsforecast  # Jupyter-only


# --- code cell ---

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# Set the environment variable to adopt the new behavior
os.environ["NIXTLA_ID_AS_COL"] = "1"

# Create a sample time series dataset
np.random.seed(42)
dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk

df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})

# Initialize the StatsForecast model
models = [AutoARIMA(season_length=7)]
sf = StatsForecast(models=models, freq="D", n_jobs=-1)

# Fit the model and generate forecasts
horizon = 14
forecasts = sf.forecast(df=df, h=horizon)

print("Forecasts:")
print(forecasts.head())

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], label="Historical Data")

# Check if 'unique_id' is in the columns (new behavior)
if "unique_id" in forecasts.columns:
    forecasts = forecasts[forecasts["unique_id"] == "series1"]

plt.plot(forecasts["ds"], forecasts["AutoARIMA"], label="Forecast", color="red")
plt.title("Time Series Forecast with AutoARIMA")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.savefig("AutoARIMA.png")
plt.show()

# Calculate and print forecast metrics
actual_values = df.iloc[-horizon:]
forecast_values = forecasts["AutoARIMA"]

mse = np.mean((actual_values["y"].values - forecast_values.values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual_values["y"].values - forecast_values.values))

print("\nForecast Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


# --- code cell ---

from datetime import datetime

import numpy as np
import pandas as pd
import requests
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA


def fetch_fred_data(series_id, api_key, start_date="2000-01-01"):
    """Fetch data from FRED API."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": datetime.now().strftime("%Y-%m-%d"),
    }
    url = "https://api.stlouisfed.org/fred/series/observations"
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        observations = data["observations"]
        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(
            df["value"], errors="coerce"
        )  # Convert to numeric, setting invalid values to NaN
        df = df.dropna(
            subset=["value"]
        )  # Remove rows with NaN values in 'value' column
        df = df.sort_values("date")
        df = df[["date", "value"]]  # Keep only 'date' and 'value' columns
        df["unique_id"] = 0  # Add unique_id column
        return df
    else:
        raise Exception(f"API request failed with status code {response.status_code}")


# Fetch data from FRED
series_id = "GDPC1"  # Real GDP series
df = fetch_fred_data(series_id, api_key)

# Rename columns to match StatsForecast expectations
df = df.rename(columns={"date": "ds", "value": "y"})

# Initialize the StatsForecast model
models = [AutoARIMA(season_length=4)]  # Assuming quarterly data for GDP
sf = StatsForecast(models=models, freq="Q", n_jobs=-1)

# Fit the model and generate forecasts
sf.fit(df)
forecasts = sf.forecast(h=4, df=df)  # Forecast the next 4 quarters
print(forecasts.head())


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast

# Assuming df is your FRED data DataFrame
# If it's not already in the correct format, you might need to preprocess it
df = df.rename(columns={"date": "ds", "value": "y"})
df["unique_id"] = "GDPC1"  # Add a unique identifier for the series

# Initialize the MLForecast model
model = MLForecast(
    models=[LGBMRegressor()],
    freq="Q",  # Change to quarterly frequency for GDP data
    lags=[1, 2, 4],  # Adjust lags for quarterly data
    date_features=["quarter", "year"],
)

# Fit the model
model.fit(df)

# Generate forecasts
horizon = 4  # Forecast next 4 quarters
forecasts = model.predict(horizon)

print("Forecasts:")
print(forecasts.head())
print("\nForecast columns:")
print(forecasts.columns)

# Prepare data for visualization
historical_data = df
forecast_data = forecasts

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
plt.plot(
    forecast_data["ds"], forecast_data["LGBMRegressor"], label="Forecast", color="red"
)
plt.title("GDP Forecast with MLForecast (LGBMRegressor)")
plt.xlabel("Date")
plt.ylabel("GDP")
plt.legend()
plt.show()

# Calculate and print forecast metrics
actual_values = historical_data.iloc[-horizon:]
forecast_values = forecast_data["LGBMRegressor"]

mse = np.mean((actual_values["y"].values - forecast_values.values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual_values["y"].values - forecast_values.values))

print("\nForecast Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


# --- code cell ---

"""
HierarchicalForecast: Hierarchical Time Series
HierarchicalForecast provides tools for forecasting hierarchical and grouped time series, ensuring coherence across different aggregation levels.
Example: Hierarchical Forecasting
"""


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from hierarchicalforecast.methods import BottomUp
from hierarchicalforecast.models import HierarchicalForecast
from hierarchicalforecast.utils import agg_series
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast.models import AutoARIMA

# Assuming df is your FRED GDP DataFrame
# If it's not already in the correct format, you might need to preprocess it
df = df.rename(columns={"date": "ds", "value": "y"})
df["ds"] = pd.to_datetime(df["ds"])

# Create a simple hierarchy for demonstration
df["Total"] = "GDP"
df["Region"] = np.random.choice(["East", "West"], size=len(df))
df["State"] = np.random.choice(["State1", "State2", "State3", "State4"], size=len(df))
df["unique_id"] = df["State"]

# Define the hierarchy
S = {
    "GDP": ["East", "West"],
    "East": ["State1", "State2"],
    "West": ["State3", "State4"],
}

# Visualize the hierarchy
G = nx.DiGraph(S)
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=3000,
    font_size=10,
    arrows=True,
)
plt.title("Hierarchical Structure of GDP")
plt.show()

# Aggregate the series
Y_df, S_df = agg_series(df, S)

# Initialize the model
hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())

# Fit and forecast
hf_model.fit(Y_df=Y_df, S=S_df, freq="Q")  # Change to quarterly frequency
forecasts = hf_model.predict(h=4)  # Forecast 4 quarters ahead
print("Forecasts:")
print(forecasts.head())

# Plot forecasts for different levels
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
levels = ["GDP", "East", "State1"]

for i, level in enumerate(levels):
    historical = Y_df[Y_df["unique_id"] == level]
    forecast = forecasts[forecasts["unique_id"] == level]

    axs[i].plot(historical["ds"], historical["y"], label="Historical")
    axs[i].plot(forecast["ds"], forecast["y"], label="Forecast", color="red")
    axs[i].set_title(f"Forecast for {level}")
    axs[i].legend()
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("GDP")

plt.tight_layout()
plt.show()


# Evaluate model performance
def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mae, rmse


# Assuming the last 4 periods are our test set
test_periods = 4
evaluation_results = {}

for level in Y_df["unique_id"].unique():
    actual = Y_df[Y_df["unique_id"] == level]["y"].iloc[-test_periods:]
    forecast = forecasts[forecasts["unique_id"] == level]["y"]
    mae, rmse = evaluate_forecast(actual, forecast)
    evaluation_results[level] = {"MAE": mae, "RMSE": rmse}

print("\nEvaluation Results:")
print(pd.DataFrame(evaluation_results).T)


# --- code cell ---

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from hierarchicalforecast.methods import BottomUp
from hierarchicalforecast.models import HierarchicalForecast
from hierarchicalforecast.utils import agg_series
from statsforecast.models import AutoARIMA

# Load sample hierarchical data
df = pd.read_csv(
    "https://raw.githubusercontent.com/Nixtla/hierarchicalforecast<your_key_here>.csv"
)
df["ds"] = pd.to_datetime(df["ds"])

# Define the hierarchy
S = {"Total": ["A", "B", "C"], "A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1", "C2"]}

# Visualize the hierarchy
G = nx.DiGraph(S)
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=3000,
    font_size=10,
    arrows=True,
)
plt.title("Hierarchical Structure")
plt.show()

# Aggregate the series
Y_df, S_df = agg_series(df, S)

# Initialize the model
hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())

# Fit and forecast
hf_model.fit(Y_df=Y_df, S=S_df, freq="W")
forecasts = hf_model.predict(h=4)
print("Forecasts:")
print(forecasts.head())

# Plot forecasts for different levels
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
levels = ["Total", "A", "A1"]

for i, level in enumerate(levels):
    historical = Y_df[Y_df["unique_id"] == level]
    forecast = forecasts[forecasts["unique_id"] == level]

    axs[i].plot(historical["ds"], historical["y"], label="Historical")
    axs[i].plot(forecast["ds"], forecast["y"], label="Forecast", color="red")
    axs[i].set_title(f"Forecast for {level}")
    axs[i].legend()
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("Value")

plt.tight_layout()
plt.show()

# Evaluate model performance
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mae, rmse


# Assuming the last 4 periods are our test set
test_periods = 4
evaluation_results = {}

for level in Y_df["unique_id"].unique():
    actual = Y_df[Y_df["unique_id"] == level]["y"].iloc[-test_periods:]
    forecast = forecasts[forecasts["unique_id"] == level]["y"]
    mae, rmse = evaluate_forecast(actual, forecast)
    evaluation_results[level] = {"MAE": mae, "RMSE": rmse}

print("\nEvaluation Results:")
print(pd.DataFrame(evaluation_results).T)


# --- code cell ---

import hierarchicalforecast

print(hierarchicalforecast.__version__)


# --- code cell ---

import pandas as pd
from hierarchicalforecast.core import HierarchicalForecast
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp
from statsforecast.models import AutoARIMA

# Load sample hierarchical data
df = pd.read_csv(
    "https://raw.githubusercontent.com/Nixtla/hierarchicalforecast<your_key_here>.csv"
)
df["ds"] = pd.to_datetime(df["ds"])

# Define the hierarchy
S = {"Total": ["A", "B", "C"], "A": ["A1", "A2"], "B": ["B1", "B2"], "C": ["C1", "C2"]}

# Initialize the model
hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())

# Fit and forecast
hf_model.fit(df, S)
forecasts = hf_model.predict(h=4)
print("Forecasts:")
print(forecasts.head())

# Evaluate the model
evaluator = HierarchicalEvaluation(models=[hf_model], df=df, S=S)
evaluation = evaluator.evaluate(test_method="last4", metrics=["mae", "rmse"])
print("\nEvaluation Results:")
print(evaluation)

# Visualize forecasts for different levels
import matplotlib.pyplot as plt

levels = ["Total", "A", "A1"]
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

for i, level in enumerate(levels):
    historical = df[df["unique_id"] == level]
    forecast = forecasts[forecasts["unique_id"] == level]

    axs[i].plot(historical["ds"], historical["y"], label="Historical")
    axs[i].plot(forecast["ds"], forecast["y"], label="Forecast", color="red")
    axs[i].set_title(f"Forecast for {level}")
    axs[i].legend()
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("Value")

plt.tight_layout()
plt.show()


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast

# Create sample data
np.random.seed(42)
dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk
df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})

# Initialize the MLForecast model
model = MLForecast(
    models=[LGBMRegressor()],
    freq="D",
    lags=[1, 7, 14],
    date_features=["dayofweek", "day"],
)

# Fit the model
model.fit(df)

# Generate forecasts
horizon = 14
forecasts = model.predict(horizon)

print("Forecasts:")
print(forecasts.head())
print("\nForecast columns:")
print(forecasts.columns)

# Prepare data for visualization
historical_data = df[df["unique_id"] == "series1"]
forecast_data = forecasts[forecasts["unique_id"] == "series1"]

# Use the correct column name for predictions
prediction_col = "LGBMRegressor"

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
plt.plot(
    forecast_data["ds"], forecast_data[prediction_col], label="Forecast", color="red"
)
plt.title("Time Series Forecast with MLForecast (LGBMRegressor)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Calculate and print forecast metrics
actual_values = historical_data.iloc[-horizon:]
forecast_values = forecast_data[prediction_col]

mse = np.mean((actual_values["y"].values - forecast_values.values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual_values["y"].values - forecast_values.values))

print("\nForecast Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


# --- code cell ---

"""
Parallel Computing and Scalability
The Nixtla suite leverages parallel computing to speed up forecasting tasks:
n_jobs: Parameter to set the number of parallel jobs.
Dask Integration: For distributed computing across clusters.
"""

# In StatsForecast, set n_jobs to -1 to use all available CPUs
sf = StatsForecast(df=df, models=models, freq="D", n_jobs=-1)


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP

np.random.seed(42)
dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk
df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})
# Initialize the NeuralForecast model
horizon = 14
model = MLP(h=horizon, input_size=28)  # 28 days of history
nf = NeuralForecast(models=[model], freq="D")

# Fit the model
nf.fit(df)

# Generate forecasts
forecasts = nf.predict()

print("\nForecasts:")
print(forecasts.head())

# Prepare data for visualization
historical_data = df[df[id_col] == df[id_col].unique()[0]]
forecast_data = forecasts[forecasts["unique_id"] == df[id_col].unique()[0]]

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(
    historical_data[time_col], historical_data[target_col], label="Historical Data"
)
plt.plot(forecast_data["ds"], forecast_data["MLP"], label="Forecast", color="red")
plt.title("Time Series Forecast with NeuralForecast (MLP)")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Calculate and print forecast metrics
actual_values = historical_data.iloc[-horizon:]
forecast_values = forecast_data["MLP"]

mse = np.mean((actual_values[target_col].values - forecast_values.values) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(actual_values[target_col].values - forecast_values.values))

print("\nForecast Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk
df = pd.DataFrame({"ds": dates, "y": values})

# Prepare features
df["date_ordinal"] = pd.to_datetime(df["ds"]).map(lambda x: x.toordinal())

# Split the data
train, test = train_test_split(df, test_size=0.2, shuffle=False)

# Prepare X and y for the model
X_train = train[["date_ordinal"]]
y_train = train["y"]
X_test = test[["date_ordinal"]]
y_test = test["y"]

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Prepare data for future predictions
future_dates = pd.date_range(
    start=df["ds"].max() + pd.Timedelta(days=1), periods=14, freq="D"
)
future_df = pd.DataFrame({"ds": future_dates})
future_df["date_ordinal"] = pd.to_datetime(future_df["ds"]).map(lambda x: x.toordinal())
future_pred = model.predict(future_df[["date_ordinal"]])

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(train["ds"], train["y"], label="Training Data")
plt.plot(test["ds"], test["y"], label="Test Data")
plt.plot(train["ds"], train_pred, label="Training Predictions", linestyle="--")
plt.plot(test["ds"], test_pred, label="Test Predictions", linestyle="--")
plt.plot(
    future_df["ds"],
    future_pred,
    label="Future Predictions",
    linestyle="--",
    color="red",
)
plt.title("Time Series Forecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Calculate and print forecast metrics
mse = np.mean((y_test - test_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - test_pred))

print("\nForecast Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS

# Disable logging to avoid the error
pl.utilities.rank_zero.rank_zero_only.rank = -1

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk
df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})

# Set the forecast horizon
horizon = 14

# Initialize the NeuralForecast models
models = [
    MLP(h=horizon, input_size=30, trainer_kwargs={"logger": False}),
    NHITS(h=horizon, input_size=30, trainer_kwargs={"logger": False}),
]

# Create the NeuralForecast object
nf = NeuralForecast(models=models, freq="D")

# Fit the model
nf.fit(df)

# Generate forecasts
forecasts = nf.predict()

# Prepare data for visualization
historical_data = df[df["unique_id"] == df["unique_id"].unique()[0]]
forecast_data = forecasts[forecasts["unique_id"] == df["unique_id"].unique()[0]]

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
plt.plot(forecast_data["ds"], forecast_data["MLP"], label="MLP Forecast", color="red")
plt.plot(
    forecast_data["ds"], forecast_data["NHITS"], label="NHITS Forecast", color="green"
)
plt.title("Time Series Forecast with NeuralForecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Calculate and print forecast metrics
actual_values = historical_data.iloc[-horizon:]["y"].values
mlp_forecast = forecast_data["MLP"].values
nhits_forecast = forecast_data["NHITS"].values


def calculate_metrics(actual, forecast):
    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - forecast))
    return mse, rmse, mae


mlp_metrics = calculate_metrics(actual_values, mlp_forecast)
nhits_metrics = calculate_metrics(actual_values, nhits_forecast)

print("\nMLP Forecast Metrics:")
print(f"Mean Squared Error (MSE): {mlp_metrics[0]:.2f}")
print(f"Root Mean Squared Error (RMSE): {mlp_metrics[1]:.2f}")
print(f"Mean Absolute Error (MAE): {mlp_metrics[2]:.2f}")

print("\nNHITS Forecast Metrics:")
print(f"Mean Squared Error (MSE): {nhits_metrics[0]:.2f}")
print(f"Root Mean Squared Error (RMSE): {nhits_metrics[1]:.2f}")
print(f"Mean Absolute Error (MAE): {nhits_metrics[2]:.2f}")


# --- code cell ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS

# Disable logging to avoid the error
pl.utilities.rank_zero.rank_zero_only.rank = -1

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
values = np.cumsum(np.random.randn(len(dates))) + 100  # Random walk
df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})

# Set the forecast horizon
horizon = 14

# Initialize the NeuralForecast models
models = [MLP(h=horizon, input_size=30), NHITS(h=horizon, input_size=30)]

# Create the NeuralForecast object with logger set to False
nf = NeuralForecast(models=models, freq="D")

# Fit the model
nf.fit(df)

# Generate forecasts
forecasts = nf.predict()

# Prepare data for visualization
historical_data = df[df["unique_id"] == df["unique_id"].unique()[0]]
forecast_data = forecasts[forecasts["unique_id"] == df["unique_id"].unique()[0]]

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
plt.plot(forecast_data["ds"], forecast_data["MLP"], label="MLP Forecast", color="red")
plt.plot(
    forecast_data["ds"], forecast_data["NHITS"], label="NHITS Forecast", color="green"
)
plt.title("Time Series Forecast with NeuralForecast")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# Calculate and print forecast metrics
actual_values = historical_data.iloc[-horizon:]["y"].values
mlp_forecast = forecast_data["MLP"].values
nhits_forecast = forecast_data["NHITS"].values


def calculate_metrics(actual, forecast):
    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - forecast))
    return mse, rmse, mae


mlp_metrics = calculate_metrics(actual_values, mlp_forecast)
nhits_metrics = calculate_metrics(actual_values, nhits_forecast)

print("\nMLP Forecast Metrics:")
print(f"Mean Squared Error (MSE): {mlp_metrics[0]:.2f}")
print(f"Root Mean Squared Error (RMSE): {mlp_metrics[1]:.2f}")
print(f"Mean Absolute Error (MAE): {mlp_metrics[2]:.2f}")

print("\nNHITS Forecast Metrics:")
print(f"Mean Squared Error (MSE): {nhits_metrics[0]:.2f}")
print(f"Root Mean Squared Error (RMSE): {nhits_metrics[1]:.2f}")
print(f"Mean Absolute Error (MAE): {nhits_metrics[2]:.2f}")


# --- code cell ---

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pyaf.hierarchicalforecast.methods import BottomUp
from pyaf.hierarchicalforecast.models import HierarchicalForecast
from pyaf.hierarchicalforecast.utils import agg_series
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsforecast.models import AutoARIMA

# Assuming df is your FRED GDP DataFrame
# If it's not already in the correct format, you might need to preprocess it
df = df.rename(columns={"date": "ds", "value": "y"})
df["ds"] = pd.to_datetime(df["ds"])

# Create a simple hierarchy for demonstration
df["Total"] = "GDP"
df["Region"] = np.random.choice(["East", "West"], size=len(df))
df["State"] = np.random.choice(["State1", "State2", "State3", "State4"], size=len(df))
df["unique_id"] = df["State"]

# Define the hierarchy
S = {
    "GDP": ["East", "West"],
    "East": ["State1", "State2"],
    "West": ["State3", "State4"],
}

# Visualize the hierarchy
G = nx.DiGraph(S)
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color="lightblue",
    node_size=3000,
    font_size=10,
    arrows=True,
)
plt.title("Hierarchical Structure of GDP")
plt.show()

# Aggregate the series
Y_df, S_df = agg_series(df, S)

# Initialize the model
hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())

# Fit and forecast
hf_model.fit(Y_df=Y_df, S=S_df, freq="Q")  # Change to quarterly frequency
forecasts = hf_model.predict(h=4)  # Forecast 4 quarters ahead
print("Forecasts:")
print(forecasts.head())

# Plot forecasts for different levels
fig, axs = plt.subplots(3, 1, figsize=(12, 15))
levels = ["GDP", "East", "State1"]

for i, level in enumerate(levels):
    historical = Y_df[Y_df["unique_id"] == level]
    forecast = forecasts[forecasts["unique_id"] == level]

    axs[i].plot(historical["ds"], historical["y"], label="Historical")
    axs[i].plot(forecast["ds"], forecast["y"], label="Forecast", color="red")
    axs[i].set_title(f"Forecast for {level}")
    axs[i].legend()
    axs[i].set_xlabel("Date")
    axs[i].set_ylabel("GDP")

plt.tight_layout()
plt.show()


# Evaluate model performance
def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return mae, rmse


# Assuming the last 4 periods are our test set
test_periods = 4
evaluation_results = {}

for level in Y_df["unique_id"].unique():
    actual = Y_df[Y_df["unique_id"] == level]["y"].iloc[-test_periods:]
    forecast = forecasts[forecasts["unique_id"] == level]["y"]
    mae, rmse = evaluate_forecast(actual, forecast)
    evaluation_results[level] = {"MAE": mae, "RMSE": rmse}

print("\nEvaluation Results:")
print(pd.DataFrame(evaluation_results).T)


# --- code cell ---

# !pip install pyaf  # Jupyter-only
