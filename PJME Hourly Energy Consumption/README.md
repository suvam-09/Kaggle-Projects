# FORECASTING HOURLY ENERGY CONSUMPTION
The project demonstrates time series forecasting using a machine learning model XGBoost to predict energy consumption with Python.

'**Time-series data**' is a sequence of data points, measuring the same thing over time, stored in a specified time order. Time intervals at which data is recorded depends upon the task at hand. Illlustrating few examples brlow which we can relate to time series data:
- Stock market movements are recorded at every millisecond of time interval
- Similarly, data from sensors in autonomous vehicles collected and processed with zero latency
- Heart beat data being recorded at every second or say, periodic intervals
These applications imply time series data is dependent on time, however, the time interval at which data is captured varies depending upon the use cases.

Types of time-series data:
- *Univariate time series data*: The term 'univariate time series' refers to a time series that consists of single (scalar) observations recorded sequentially over equal time intervals. Say, for example, the temperature pattern follows a univariate time series data.
- *Multivariate time series data*: As the name suggests, multiple variables are recorded at given time intervals. Say, fo example, open, high, low and closing values in stock market data.

Patterns followed by a time-series data:
- Purely Random Error (no recognisable pattern)
- Curvilinear Trend (quadratic, exponential)
- Increasing/Decreasing Linear Trend
- Seasonal Pattern (ups and downs)
- Seasonal Pattern plus Linear Growth

Here, in this case we would be dealing with the seasonal pattern time-series data.

**Time series data forecasting** can be defined as predicting upcoming future values by looking at its previous recorded values at successive time intervals. Forecasting data using time-series analysis comprises the use of some significant model to forecast future conclusions on the basis of known past outcome.

In this project, we would be implementing the XGBoost algorithm to train our model. In addition to it, there are various other steps involved that we have used over our trained model to help predict the future values. Summarising these steps below:

(a). **FEATURE ENGINEERING**:
Herein, we run through the initial phase of our understanding the data and then drawing out the feature based on our dataset. For example, we segregate the hourly timeframe in dataset to build new features such as hours, day of the week, months, years, day of the year, week of the year, etc.

(b). **OUTLIER ANALYSIS**:
Visualizing the graph for the dataset, at certain timeframes, the power consumption values are extremely low compared to the overall dataset. We term these offset values as outliers. Since, such outliers will impact our training model, we follow the outlier removal rule, so as to eliminate these outliers.

(c). **FORECASTING HORIZON**:
The forecast horizon is the length of time into the future for which forecasts are to be prepared. These generally vary from short-term forecasting horizons (less than three months) to long-term horizons (more than two years).

(d). **TIME SERIES CROSS VALIDATION**:
In this process, the dataset is divided into (n) number of splits, wherein the model is trained on (n-1) set of data and the evaluation occurs on the remaining set. The procedure starts with randomly splitting the original dataset into (n) number of folds or subsets. In each iteration, the model is trained on the (n-1) subsets of the entire dataset. After that, the model is tested on the (n)th subset to check its performance.

This process is repeated until all of the n-folds have served as the evaluation set. The results of each iteration are averaged, and it's called the cross-validation accuracy. Cross-validation accuracy is used as a performance metric to compare the efficiency of different models. To help visualize, let's assume an example below, say we have 5 folds, so data is split into 5 sets (A, B, C, D, E) and then the procedure follow the cycle as depicted below:
- First iteration: Training set (A, B, C, D), Validation set (E)
- Second iteration: Training set (E, A, B, C), Validation set (D)
- Third iteration: Training set (D, E, A, B), Validation set (C)
- Fourth iteration: Training set (C, D, E, A), Validation set (B)
- Fifth iteration: Training set (B, C, D, E), Validation set (A)

(e). **LAG FEATURE**: (how far into the future do we want to predict?) - 
Herein, we are asking the model to look back in past (say, X days back) and use the target value for that many days in the past as a new feature that we feed into the model.

(f). **PREDICTING THE FUTURE**:
Herein in the final step, the model uses the lag features as a new feature for predicting the future values.
