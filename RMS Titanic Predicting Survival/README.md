# RMS TITANIC: PREDICTING SURVIVAL
Building a predictive model using passenger data (ie name, age, gender, socio-economic class, etc.) that would help answer the question: “*What sort of people were more likely to survive on RMS Titanic?*”.

Few valuable insights as part of Exploratory Data Analysis (EDA) have been shared below:
1. Percentage of survivors is greater in case of passengers holding 'Class 1' ticket compared to the ones holding 'Class 3' ticket, as they were provided first preference in terms of access to lifejackets and aboarding the lifeboats.
2. Significant differences in survival rates summarizes that the guests on the upper decks were provided first preference in terms of access to lifejackets and aboarding the lifeboats, compared to the passengers who were aboard the lower decks.
3. Percentage of survivors is greater in case of female passengers compared to the male passengers. This summarizes that the female passengers/children were provided preference in terms of access to lifejackets and aboarding the lifeboats, as compared to male passengers.

Post these analyses we go ahead with the usual process as decribed below:
1. Feature Engineering
2. Handling Outliers
3. One Hot Encoding (transforming the categorical features to dummy variables)
4. Performing Train/Test Split
5. Evaluating the Features Importance
6. Applying the Machine Learning Models

The **XGBClassifier** model performs fairly well compared to the others, with an accuracy score of  *83.41*. However, there's more we could do here to improve our accuracy score. Say for example, we could evaluate the error metric and go ahead performing the tuning of the hyperparameters using K-fold cross validation to improve the accuracy score. Or else, we can consider dropping the features which do not hold any significance in predicting the survival rates, and try re-running our model on the test dataset. 