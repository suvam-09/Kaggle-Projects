# Boston House Price Prediction
The project highlights an applied usage of machine learning focused on predicting house prices. The dataset used in this project comes from the UCI Machine Learning Repository. This data was collected in 1978 and each of the 506 entries represents aggregate information about 14 features of homes from various suburbs located in Boston. The dataset is available on Kaggle. Our goal is to develop a model that has the capacity of predicting the value of houses.

The features can be summarized as follows:
- *CRIM*: This is the per capita crime rate by town
- *ZN*: This is the proportion of residential land zoned for lots larger than 25000 sq.ft.
- *INDUS*: This is the proportion of non-retail business acres per town.
- *CHAS*: This is the Charles River dummy variable (this is equal to 1 if tract bounds river; 0 otherwise)
- *NOX*: This is the nitric oxides concentration (parts per 10 million)
- *RM*: This is the average number of rooms per dwelling
- *AGE*: This is the proportion of owner-occupied units built prior to 1940
- *DIS*: This is the weighted distances to five Boston employment centers
- *RAD*: This is the index of accessibility to radial highways
- *TAX*: This is the full-value property-tax rate per $10000
- *PTRATIO*: This is the pupil-teacher ratio by town
- *B*: This is calculated as 1000(Bk — 0.63)², where Bk is the proportion of people of African-American descent by town
- *LSTAT*: This is the percentage lower status of the population
- *MEDV*: This is the median value of owner-occupied homes in $1000s

Various machine learning models such as Linear Regression, Lasso Regression, ElasticNet Regression & Xtreme Gradient Boosting Regression have been implemented over the preprocessed data to evaluate the model with the highest accuracy. Accuracy for the model are defined using performance metrics which includes calculating various types of error, the goodness of fit, or some other useful measurement. For this project I have used the coefficient of determination (R²) to quantify the model’s performance. The *coefficient of determination* for a model is a useful statistic in regression analysis as it often describes how "good" that model is at making predictions. Please refer below table with regards to the performance metrics for each of the machine learning models that have been implemented in the project:



Click here to run the application live on server

## App Features:
- **Step 1**: User needs to input the parameters
- **Step 2**: These parameters are stored as a dataframe and displayed on the screen
- **Step 3**: The input paramters are fed to our model which displays the predicted median price
- **Step 4**: Displaying the plots for feature importance based on SHAP values related to the prediction