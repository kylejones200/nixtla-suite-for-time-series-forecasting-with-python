# Nixtla Suite for Time Series Forecasting with Python

This project demonstrates time series forecasting using the Nixtla StatsForecast library with AutoARIMA.

## Business context

The Nixtla suite is a collection of Python libraries for time series analysis. It feels like something that will be easier to use with more practice. Having used a lot of libraries this is the one that haunts my thoughts --- "I wonder if I could do this with NeuralForecast? ..." But in the end, I struggled to get NeuralForecast to do what I wanted with N-BEATS --- a task that was easy with DARTS.

Nixtla Suite is made of several libraries, each targeting specific forecasting need:

- `StatsForecast`: Efficient implementations of statistical models. - `NeuralForecast`: Deep learning models for time series forecasting. - `HierarchicalForecast`: Tools for hierarchical and grouped time series. - `MLForecast`: Machine learning models tailored for time series data.

## Article

Medium article: [Nixtla Suite for Time Series Forecasting](https://medium.com/@kylejones_47003/nixtla-suite-for-time-series-forecasting-with-python-b0f318365e9b)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files (if needed)
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data generation parameters (start date, periods, frequency)
- Model parameters (season length, frequency)
- Forecast horizon
- Output settings

## Caveats

- By default, the script generates synthetic ERCOT-style hourly data.
- The model uses AutoARIMA with seasonal length of 24 hours.
- StatsForecast requires the `NIXTLA_ID_AS_COL` environment variable to be set.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).