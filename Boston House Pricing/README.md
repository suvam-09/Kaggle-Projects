# BOSTON HOUSE PRICE PREDICTION
The project highlights a streamlit web application demonstrating the applied usage of machine learning focused on predicting house prices. The user interface allows the users to enter the values for features of a house as an input based on which the median price is calculated and displayed back to the user.

-> **Click [here](https://blink-house-price-predictor.herokuapp.com/) to run the application live on server**

<img src = ".\Images\image_4.jpg">

## OVERVIEW
The dataset used in this project comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston. The dataset is also available on [Kaggle](https://www.kaggle.com/datasets/jamieleech/boston-housing-dataset). The features can be summarized as follows:
- **CRIM**: This denotes the per capita crime rate by town
- **ZN**: This denotes the proportion of residential land zoned for lots larger than 25000 sq.ft.
- **INDUS**: This denotes the proportion of non-retail business acres per town.
- **CHAS**: This denotes the Charles River dummy variable (this is equal to 1 if tract bounds river; 0 otherwise)
- **NOX**: This denotes the nitric oxides concentration (parts per 10 million)
- **RM**: This denotes the average number of rooms per dwelling
- **AGE**: This denotes the proportion of owner-occupied units built prior to 1940
- **DIS**: This denotes the weighted distances to five Boston employment centers
- **RAD**: This denotes the index of accessibility to radial highways
- **TAX**: This denotes the full-value property-tax rate per $10000
- **PTRATIO**: This denotes the pupil-teacher ratio by town
- **B**: This denotes calculated as 1000(Bk — 0.63)², where Bk is the proportion of people of African-American descent by town
- **LSTAT**: This denotes the percentage lower status of the population
- **MEDV**: This denotes the median value of owner-occupied homes in $1000s

The goal is to develop a model that has the capacity of predicting the price of houses (i.e. *MEDV*). Different machine learning models such as Linear Regression, Lasso Regression, ElasticNet Regression & Extreme Gradient Boosting Regression have been implemented over the preprocessed data to evaluate the model with the highest accuracy. The accuracy for a model is defined using performance metrics which includes calculating various types of error, the goodness of fit, or some other useful measurement. In this project I have considered *coefficient of determination* (R²) to quantify the model’s performance. The *coefficient of determination* for a model is a useful statistic in regression analysis as it often describes how "good" that model is at making predictions. Please refer below table with regards to the performance metrics for each of the machine learning models that have been implemented in the project:

<img src = ".\Images\screenshot_3.PNG">

** (**model accuracy = R-squared error * 100**)

## SCREENSHOTS

- Below screenshot illustrates the application's basic interface that is viewable to the users once they are directed to the application. If we observe on the left hand side, there is a sidebar wherein the user can drag the slider either left or right for each of the features. These values serve as an input for the application and the corresponding values are displayed as a dataframe over the right side of the sidebar. By default, the slider is set at the *mean value* for each of the features. So whenever we hit the link for the application (or) the application is reloaded, the slider will point to the mean value unless the user modifies the values by dragging the slider to left or right.

<img src = ".\Images\screenshot_1.PNG">

- Below screenshot illustrates the feature importance plot denoting the correlation for each of these features against the median pricing. Once the values have been provided by the user, the application displays the median price value based on the trained model. Underneath the predicted price section, the application displays a graph visualizing how each of these features correlate with the target variable (i.e. *median price*). The red color on the color scale indicates *higher correlation* while the blue color indicates *lower correlation*.
> *Considering the example for **LSTAT** feature, the red color is significant onto the negative side of the scale. This indicates that the LSTAT feature has a greater negative effect compared to positive effect in predicting the median price. In simple terms, both are inversely proportional to each other which means that **an increase in the LSTAT value would lead to a large significant drop in median price value and vice-versa**. While for **RM** feature which denotes average number of rooms, the red color is significant towards the positive side of the scale. This implies that the RM feature has a greater positive effect compared to negative effect in predicting the median price. So **an increase in the value for RM would lead to a significant increase in median price value and vice-versa***.

<img src = ".\Images\screenshot_2.PNG">

## LIMITATION:
It is not possible if a customer wants to predict the future price of the house because there is a risk as the prices of an area increases continuously. So basically the price of a plot of land or a house is a continuous variable in today's date which makes its challenging to have it forecasted using some machine learning model. So the customers tend to hire a broker or some real estate agent to minimize this error but again, the cost of the entire process increases.
