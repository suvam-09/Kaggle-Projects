# Boston House Price Prediction
The project highlights a stremalit web application demonstrating the applied usage of machine learning focused on predicting house prices. The application's interface allows the users to share their parameters respective to a house as an input based on which the median price for that house is calculated and displayed back to the user.

Click [here](https://blink-house-price-predictor.herokuapp.com/) to run the application live on server

<img src = ".\Images\image_4.jpg">

## Overview
The dataset used in this project comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston. The dataset is also available on [Kaggle](). The features can be summarized as follows:
- **CRIM**: This is the per capita crime rate by town
- **ZN**: This is the proportion of residential land zoned for lots larger than 25000 sq.ft.
- **INDUS**: This is the proportion of non-retail business acres per town.
- **CHAS**: This is the Charles River dummy variable (this is equal to 1 if tract bounds river; 0 otherwise)
- **NOX**: This is the nitric oxides concentration (parts per 10 million)
- **RM**: This is the average number of rooms per dwelling
- **AGE**: This is the proportion of owner-occupied units built prior to 1940
- **DIS**: This is the weighted distances to five Boston employment centers
- **RAD**: This is the index of accessibility to radial highways
- **TAX**: This is the full-value property-tax rate per $10000
- **PTRATIO**: This is the pupil-teacher ratio by town
- **B**: This is calculated as 1000(Bk — 0.63)², where Bk is the proportion of people of African-American descent by town
- **LSTAT**: This is the percentage lower status of the population
- **MEDV**: This is the median value of owner-occupied homes in $1000s

The goal is to develop a model that has the capacity of predicting the value of houses (i.e. *MEDV*). Various machine learning models such as Linear Regression, Lasso Regression, ElasticNet Regression & Xtreme Gradient Boosting Regression have been implemented over the preprocessed data to evaluate the model with the highest accuracy. The accuracy for a model is defined using performance metrics which includes calculating various types of error, the goodness of fit, or some other useful measurement. In this project I have considered *coefficient of determination* (R²) to quantify the model’s performance. The *coefficient of determination* for a model is a useful statistic in regression analysis as it often describes how "good" that model is at making predictions. Please refer below table with regards to the performance metrics for each of the machine learning models that have been implemented in the project:

<img src = ".\Images\screenshot_3.PNG">

*model accuracy = R-squared error * 100*

## Screenshots

Below screenshot is the application's basic interface that is viewable to the users once they are directed to the application. If we observe on the left hand side, there is a sidebar wherein the user can drag the slider either left or right for each of the features. These values serve as an input for the application and the corresponding values are displayed as a dataframe onto the right side of the sidebar.

<img src = ".\Images\screenshot_1.PNG">

Once the values have been provided by the user, the application displays the median price value and just below the predicted price section, the application displays a graph visualizing how each of these features correlate with the target variable (i.e. *median price*). If we analyze the screenshot provided below, the red color on the color scale indicates *higher correlation* while the blue color indicates *lower correlation*. Considering the example for *LSTAT* feature, the red color is significant onto the negative side. This means that the median price has a high negative correlation with the LSTAT feature. In simple terms, both are *inversely proportional*- this means, if we increase the value for LSTAT it would lead to a significant drop in median price value and vice-versa.

<img src = ".\Images\screenshot_2.PNG">