# Nixtla Suite for Time Series Forecasting with Python Nixtla brings together multiple specialized libraries for time series
analysis, from traditional statistical models to advanced neural...

### Nixtla Suite for Time Series Forecasting with Python
#### Nixtla brings together multiple specialized libraries for time series analysis, from traditional statistical models to advanced neural networks. Its modular approach allows users to choose the right tool for their forecasting needs, whether it's high-performance statistical models, machine learning algorithms, or deep learning architectures.
The **Nixtla suite** is a collection of Python libraries for time series analysis. It feels like something that will be easier to use with more practice. Having used a lot of libraries this is the one that haunts my thoughts --- "I wonder if I could do this with NeuralForecast? ..." But in the end, I struggled to get NeuralForecast to do what I wanted with N-BEATS --- a task that was easy with DARTS.

Nixtla Suite is made of several libraries, each targeting specific forecasting need:

- `StatsForecast`: Efficient implementations of statistical models.
- `NeuralForecast`: Deep learning models for time series forecasting.
- `HierarchicalForecast`: Tools for hierarchical and grouped time series.
- `MLForecast`: Machine learning models tailored for time series data.

Installation: You can install the Nixtla suite libraries individually using `pip`:


### StatsForecast: High-Performance Statistical Models
**StatsForecast** provides fast implementations of classic statistical models like ARIMA, ETS, and more. Let's do a simple for forecast with autoARIMA.


### MLForecast: Machine Learning for Time Series
**MLForecast** simplifies the application of machine learning models to time series data by automating feature creation and model training. Let's try forecasting with LightGBM (version 1). This version uses data from FRED.


Give the same simulated data, LGBMRegressor did much better than a basic regression using sklearn.


I was excited to try the hierarchical and grouped time series but I couldn't get the HierarchicalForecast to work.

**Real world example: ERCOT Energy Load Data**

Back to StatsForcast. I wanted to try it with a different dataset. So in the examples below, I use energy load data from ERCOT, the grid balancing authority in Texas.




Let's dive a little deeper.





<figcaption>Visualization from StatForecast of ERCOT data</figcaption>




This is pretty cool. We can see how well each of these models does for our dataset based on MAPE. Dynamic Optimized Theta wins!



What about the LightGBM?



By comparison, ARIMA was MAE of 11.

#### So what?
I feel like I've only scratched the surface of what the Nixtla suite can do. I think I could make the analysis much faster for large datasets by parallelizing using Nixtla. It is a little finicky and not as straightforward as something like statsmodels but it has more capabilities.
