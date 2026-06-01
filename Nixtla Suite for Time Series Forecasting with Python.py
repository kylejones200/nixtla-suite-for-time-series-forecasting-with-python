"""Generated from Jupyter notebook: Nixtla Suite for Time Series Forecasting with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import logging
import os
from datetime import datetime

import hierarchicalforecast
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from hierarchicalforecast.core import HierarchicalForecast
from hierarchicalforecast.evaluation import HierarchicalEvaluation
from hierarchicalforecast.methods import BottomUp
from hierarchicalforecast.models import HierarchicalForecast
from hierarchicalforecast.utils import agg_series
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NHITS
from pyaf.hierarchicalforecast.methods import BottomUp
from pyaf.hierarchicalforecast.models import HierarchicalForecast
from pyaf.hierarchicalforecast.utils import agg_series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

logger = logging.getLogger(__name__)
def calculate_metrics(actual, forecast):
    mse = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - forecast))
    return (mse, rmse, mae)


def evaluate_forecast(actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return (mae, rmse)


def fetch_fred_data(series_id, start_date="2000-01-01", end_date=None):
    """Fetch data from FRED via pandas_datareader."""
    import pandas_datareader.data as web

    if end_date is None:
        end_date = datetime.now()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    raw = web.DataReader(series_id, "fred", start, end)
    df = raw.reset_index()
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[-1]
    out = df.rename(columns={date_col: "date", value_col: "value"})
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"])


def fetch_fred_data_nixtla(series_id, start_date="2000-01-01", end_date=None):
    df = fetch_fred_data(series_id, start_date, end_date)
    df["unique_id"] = 0
    return df


def set_the_environment_variable_to_adopt_the_new_be() -> None:
    os.environ["NIXTLA_ID_AS_COL"] = "1"
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
    values = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})
    models = [AutoARIMA(season_length=7)]
    sf = StatsForecast(models=models, freq="D", n_jobs=-1)
    horizon = 14
    forecasts = sf.forecast(df=df, h=horizon)
    print("Forecasts:")
    print(forecasts.head())
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["y"], label="Historical Data")
    if "unique_id" in forecasts.columns:
        forecasts = forecasts[forecasts["unique_id"] == "series1"]

    plt.plot(forecasts["ds"], forecasts["AutoARIMA"], label="Forecast", color="red")
    plt.title("Time Series Forecast with AutoARIMA")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("AutoARIMA.png")
    plt.show()
    actual_values = df.iloc[-horizon:]
    forecast_values = forecasts["AutoARIMA"]
    mse = np.mean((actual_values["y"].values - forecast_values.values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values["y"].values - forecast_values.values))
    print("\nForecast Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def fetch_data_from_fred() -> None:
    series_id = "GDPC1"
    df = fetch_fred_data_nixtla(series_id)
    df = df.rename(columns={"date": "ds", "value": "y"})
    models = [AutoARIMA(season_length=4)]
    sf = StatsForecast(models=models, freq="Q", n_jobs=-1)
    sf.fit(df)
    forecasts = sf.forecast(h=4, df=df)
    print(forecasts.head())


def assuming_df_is_your_fred_data_dataframe() -> None:
    df = df.rename(columns={"date": "ds", "value": "y"})
    df["unique_id"] = "GDPC1"
    model = MLForecast(
        models=[LGBMRegressor()],
        freq="Q",
        lags=[1, 2, 4],
        date_features=["quarter", "year"],
    )
    model.fit(df)
    horizon = 4
    forecasts = model.predict(horizon)
    print("Forecasts:")
    print(forecasts.head())
    print("\nForecast columns:")
    print(forecasts.columns)
    historical_data = df
    forecast_data = forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
    plt.plot(
        forecast_data["ds"],
        forecast_data["LGBMRegressor"],
        label="Forecast",
        color="red",
    )
    plt.title("GDP Forecast with MLForecast (LGBMRegressor)")
    plt.xlabel("Date")
    plt.ylabel("GDP")
    plt.legend()
    plt.show()
    actual_values = historical_data.iloc[-horizon:]
    forecast_values = forecast_data["LGBMRegressor"]
    mse = np.mean((actual_values["y"].values - forecast_values.values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values["y"].values - forecast_values.values))
    print("\nForecast Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def assuming_df_is_your_fred_gdp_dataframe() -> None:
    "\nHierarchicalForecast: Hierarchical Time\xa0Series\nHierarchicalForecast provides tools for forecasting hierarchical and grouped time series, ensuring coherence across different aggregation levels.\nExample: Hierarchical Forecasting\n"
    df = df.rename(columns={"date": "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["Total"] = "GDP"
    df["Region"] = np.random.choice(["East", "West"], size=len(df))
    df["State"] = np.random.choice(
        ["State1", "State2", "State3", "State4"], size=len(df)
    )
    df["unique_id"] = df["State"]
    S = {
        "GDP": ["East", "West"],
        "East": ["State1", "State2"],
        "West": ["State3", "State4"],
    }
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
    Y_df, S_df = agg_series(df, S)
    hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())
    hf_model.fit(Y_df=Y_df, S=S_df, freq="Q")
    forecasts = hf_model.predict(h=4)
    print("Forecasts:")
    print(forecasts.head())
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
    test_periods = 4
    evaluation_results = {}
    for level in Y_df["unique_id"].unique():
        actual = Y_df[Y_df["unique_id"] == level]["y"].iloc[-test_periods:]
        forecast = forecasts[forecasts["unique_id"] == level]["y"]
        mae, rmse = evaluate_forecast(actual, forecast)
        evaluation_results[level] = {"MAE": mae, "RMSE": rmse}

    print("\nEvaluation Results:")
    print(pd.DataFrame(evaluation_results).T)


def load_sample_hierarchical_data() -> None:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Nixtla/hierarchicalforecast<your_key_here>.csv"
    )
    df["ds"] = pd.to_datetime(df["ds"])
    S = {
        "Total": ["A", "B", "C"],
        "A": ["A1", "A2"],
        "B": ["B1", "B2"],
        "C": ["C1", "C2"],
    }
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
    Y_df, S_df = agg_series(df, S)
    hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())
    hf_model.fit(Y_df=Y_df, S=S_df, freq="W")
    forecasts = hf_model.predict(h=4)
    print("Forecasts:")
    print(forecasts.head())
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
    test_periods = 4
    evaluation_results = {}
    for level in Y_df["unique_id"].unique():
        actual = Y_df[Y_df["unique_id"] == level]["y"].iloc[-test_periods:]
        forecast = forecasts[forecasts["unique_id"] == level]["y"]
        mae, rmse = evaluate_forecast(actual, forecast)
        evaluation_results[level] = {"MAE": mae, "RMSE": rmse}

    print("\nEvaluation Results:")
    print(pd.DataFrame(evaluation_results).T)


def notebook_step_007() -> None:
    print(hierarchicalforecast.__version__)


def load_sample_hierarchical_data_2() -> None:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Nixtla/hierarchicalforecast<your_key_here>.csv"
    )
    df["ds"] = pd.to_datetime(df["ds"])
    S = {
        "Total": ["A", "B", "C"],
        "A": ["A1", "A2"],
        "B": ["B1", "B2"],
        "C": ["C1", "C2"],
    }
    hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())
    hf_model.fit(df, S)
    forecasts = hf_model.predict(h=4)
    print("Forecasts:")
    print(forecasts.head())
    evaluator = HierarchicalEvaluation(models=[hf_model], df=df, S=S)
    evaluation = evaluator.evaluate(test_method="last4", metrics=["mae", "rmse"])
    print("\nEvaluation Results:")
    print(evaluation)
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


def create_sample_data() -> None:
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
    values = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})
    model = MLForecast(
        models=[LGBMRegressor()],
        freq="D",
        lags=[1, 7, 14],
        date_features=["dayofweek", "day"],
    )
    model.fit(df)
    horizon = 14
    forecasts = model.predict(horizon)
    print("Forecasts:")
    print(forecasts.head())
    print("\nForecast columns:")
    print(forecasts.columns)
    historical_data = df[df["unique_id"] == "series1"]
    forecast_data = forecasts[forecasts["unique_id"] == "series1"]
    prediction_col = "LGBMRegressor"
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
    plt.plot(
        forecast_data["ds"],
        forecast_data[prediction_col],
        label="Forecast",
        color="red",
    )
    plt.title("Time Series Forecast with MLForecast (LGBMRegressor)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    actual_values = historical_data.iloc[-horizon:]
    forecast_values = forecast_data[prediction_col]
    mse = np.mean((actual_values["y"].values - forecast_values.values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values["y"].values - forecast_values.values))
    print("\nForecast Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def in_statsforecast_set_n_jobs_to_1_to_use_all_avai() -> None:
    "\nParallel Computing and Scalability\nThe Nixtla suite leverages parallel computing to speed up forecasting tasks:\nn_jobs: Parameter to set the number of parallel jobs.\nDask Integration: For distributed computing across clusters.\n"
    StatsForecast(df=df, models=models, freq="D", n_jobs=-1)


def initialize_the_neuralforecast_model() -> None:
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
    values = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})
    horizon = 14
    model = MLP(h=horizon, input_size=28)
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df)
    forecasts = nf.predict()
    print("\nForecasts:")
    print(forecasts.head())
    historical_data = df[df[id_col] == df[id_col].unique()[0]]
    forecast_data = forecasts[forecasts["unique_id"] == df[id_col].unique()[0]]
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
    actual_values = historical_data.iloc[-horizon:]
    forecast_values = forecast_data["MLP"]
    mse = np.mean((actual_values[target_col].values - forecast_values.values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values[target_col].values - forecast_values.values))
    print("\nForecast Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def generate_sample_data() -> None:
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
    values = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({"ds": dates, "y": values})
    df["date_ordinal"] = pd.to_datetime(df["ds"]).map(lambda x: x.toordinal())
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    X_train = train[["date_ordinal"]]
    y_train = train["y"]
    X_test = test[["date_ordinal"]]
    y_test = test["y"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    future_dates = pd.date_range(
        start=df["ds"].max() + pd.Timedelta(days=1), periods=14, freq="D"
    )
    future_df = pd.DataFrame({"ds": future_dates})
    future_df["date_ordinal"] = pd.to_datetime(future_df["ds"]).map(
        lambda x: x.toordinal()
    )
    future_pred = model.predict(future_df[["date_ordinal"]])
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
    mse = np.mean((y_test - test_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - test_pred))
    print("\nForecast Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")


def disable_logging_to_avoid_the_error() -> None:
    pl.utilities.rank_zero.rank_zero_only.rank = -1
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
    values = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})
    horizon = 14
    models = [
        MLP(h=horizon, input_size=30, trainer_kwargs={"logger": False}),
        NHITS(h=horizon, input_size=30, trainer_kwargs={"logger": False}),
    ]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df)
    forecasts = nf.predict()
    historical_data = df[df["unique_id"] == df["unique_id"].unique()[0]]
    forecast_data = forecasts[forecasts["unique_id"] == df["unique_id"].unique()[0]]
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
    plt.plot(
        forecast_data["ds"], forecast_data["MLP"], label="MLP Forecast", color="red"
    )
    plt.plot(
        forecast_data["ds"],
        forecast_data["NHITS"],
        label="NHITS Forecast",
        color="green",
    )
    plt.title("Time Series Forecast with NeuralForecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    actual_values = historical_data.iloc[-horizon:]["y"].values
    mlp_forecast = forecast_data["MLP"].values
    nhits_forecast = forecast_data["NHITS"].values
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


def disable_logging_to_avoid_the_error_2() -> None:
    pl.utilities.rank_zero.rank_zero_only.rank = -1
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2022-12-31", freq="D")
    values = np.cumsum(np.random.randn(len(dates))) + 100
    df = pd.DataFrame({"unique_id": "series1", "ds": dates, "y": values})
    horizon = 14
    models = [MLP(h=horizon, input_size=30), NHITS(h=horizon, input_size=30)]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df)
    forecasts = nf.predict()
    historical_data = df[df["unique_id"] == df["unique_id"].unique()[0]]
    forecast_data = forecasts[forecasts["unique_id"] == df["unique_id"].unique()[0]]
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data["ds"], historical_data["y"], label="Historical Data")
    plt.plot(
        forecast_data["ds"], forecast_data["MLP"], label="MLP Forecast", color="red"
    )
    plt.plot(
        forecast_data["ds"],
        forecast_data["NHITS"],
        label="NHITS Forecast",
        color="green",
    )
    plt.title("Time Series Forecast with NeuralForecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    actual_values = historical_data.iloc[-horizon:]["y"].values
    mlp_forecast = forecast_data["MLP"].values
    nhits_forecast = forecast_data["NHITS"].values
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


def assuming_df_is_your_fred_gdp_dataframe_2() -> None:
    df = df.rename(columns={"date": "ds", "value": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["Total"] = "GDP"
    df["Region"] = np.random.choice(["East", "West"], size=len(df))
    df["State"] = np.random.choice(
        ["State1", "State2", "State3", "State4"], size=len(df)
    )
    df["unique_id"] = df["State"]
    S = {
        "GDP": ["East", "West"],
        "East": ["State1", "State2"],
        "West": ["State3", "State4"],
    }
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
    Y_df, S_df = agg_series(df, S)
    hf_model = HierarchicalForecast(model=AutoARIMA(), reconciliation=BottomUp())
    hf_model.fit(Y_df=Y_df, S=S_df, freq="Q")
    forecasts = hf_model.predict(h=4)
    print("Forecasts:")
    print(forecasts.head())
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
    test_periods = 4
    evaluation_results = {}
    for level in Y_df["unique_id"].unique():
        actual = Y_df[Y_df["unique_id"] == level]["y"].iloc[-test_periods:]
        forecast = forecasts[forecasts["unique_id"] == level]["y"]
        mae, rmse = evaluate_forecast(actual, forecast)
        evaluation_results[level] = {"MAE": mae, "RMSE": rmse}

    print("\nEvaluation Results:")
    print(pd.DataFrame(evaluation_results).T)


def main() -> None:
    set_the_environment_variable_to_adopt_the_new_be()
    fetch_data_from_fred()
    assuming_df_is_your_fred_data_dataframe()
    assuming_df_is_your_fred_gdp_dataframe()
    load_sample_hierarchical_data()
    notebook_step_007()
    load_sample_hierarchical_data_2()
    create_sample_data()
    in_statsforecast_set_n_jobs_to_1_to_use_all_avai()
    initialize_the_neuralforecast_model()
    generate_sample_data()
    disable_logging_to_avoid_the_error()
    disable_logging_to_avoid_the_error_2()
    assuming_df_is_your_fred_gdp_dataframe_2()


if __name__ == "__main__":
    main()
