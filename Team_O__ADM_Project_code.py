# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from tensorflow.contrib import skflow
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor




# Fetching Dataset

bike_sharing_data = pd.read_csv("bike_sharing_daily.csv")

print "Data read successfully!"


# Extracting

feature_columns = bike_sharing_data.columns[:-3]  # all columns but last are features
target_column = bike_sharing_data.columns[-1]  # last column is the target

print ("Feature column(s):\n{}\n".format(feature_cols))
print ("Target column:\n{}".format(target_col))

# Exploration

print {"\n Data values:"}
print bike_sharing_data.head()  # print the first 5 rows

print {"\n Data stats:"}
bike_sharing_data.describe() # shows stats

print("\n Dimensions") # dimensions of dataset(rows,columns)
bike_sharing_data.shape

print("\n Number of values per each range of month")
bike_sharing_data['mnth'].value_counts()# counts the number of values per each range

print("\n Null Values in the data")
bike_sharing_data.isnull().sum() #show null values

# Exploratory Visulazation

plt.style.use('ggplot')

bike_sharing_data.boxplot(column='cnt', by=['yr','mnth'])

plt.title('Number of bikes rented per month')
plt.xlabel('')
plt.xticks((np.arange(0,len(bike_sharing_data)/30,len(bike_sharing_data)/731)), calendar.month_name[1:13]*2, rotation=45)
plt.ylabel('Number of bikes')

plt.show()


#Data Preprocessing (Methodology)

# Pre-processing

X = bike_sharing_data[feature_columns.drop(['dteday'],['instant'])] # feature values 
y = bike_sharing_data[target_column]  # corresponding targets

#Algorithms and Techniques
# Split

X_train, X_test, y_train, y_test = train_test_split(X, y)# test size is set to 0.25


# Training Algorithms
svr = SVR()
lr = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
gbr = GradientBoostingRegressor()


fit1 = lr.fit(X_train,y_train)#Here we fit training data to linear regressor
fit2 = dtr.fit(X_train,y_train)#Here we fit training data to Decision Tree Regressor
fit3 = rfr.fit(X_train,y_train)#Here we fit training data to Random Forest Regressor
fit4 = gbr.fit(X_train,y_train)#Here we fit training data to Gradient Boosting Regressor
fit5 = svr.fit(X_train, y_train)#Here we fit training data to Support Vector Regressor

print("Accuracy Score of Linear regression on train set",fit1.score(X_train,y_train)*100)
print("Accuracy Score of Decision Tree on train set",fit2.score(X_train,y_train)*100)
print("Accuracy Score of Random Forests on train set",fit3.score(X_train,y_train)*100)
print("Accuracy Score of Gradient Boosting on train set",fit4.score(X_train,y_train)*100)
print("Accuracy Score of SVR on train set",fit5.score(X_train,y_train)*100)


# Validation

# Validation Linear Regression

lr_pred = lr.predict(X_test)

score_lr = r2_score(y_test, lr_pred)
rmse_lr = sqrt(mean_squared_error(y_test, lr_pred))

print("Score LR: %f" % score_lr)
print("RMSE LR: %f" % rmse_lr)

# Validtion Decision Tree

dtr_pred = dtr.predict(X_test)

score_dtr = r2_score(y_test, dtr_pred)
rmse_dtr = sqrt(mean_squared_error(y_test, dtr_pred))

print("Score DTR: %f" % score_dtr)
print("RMSE DTR: %f" % rmse_dtr)

# Validation Random Forest Regressor

rfr_pred = rfr.predict(X_test)

score_rfr = r2_score(y_test, rfr_pred)
rmse_rfr = sqrt(mean_squared_error(y_test, rfr_pred))

print("Score RFR: %f" % score_rfr)
print("RMSE RFR: %f" % rmse_rfr)

# Validation Gradient Boosting Regressor

gbr_pred = gbr.predict(X_test)

score_gbr = r2_score(y_test, gbr_pred)
rmse_gbr = sqrt(mean_squared_error(y_test, gbr_pred))

print("Score GBR: %f" % score_gbr)
print("RMSE GBR: %f" % rmse_gbr)

# Validation SVR

svr_pred = svr.predict(X_test)

score_svr = r2_score(y_test, svr_pred)
rmse_svr = sqrt(mean_squared_error(y_test, svr_pred))

print("Score SVR: %f" % score_svr)
print("RMSE SVR: %f" % rmse_svr)


#Methodology


# Tuning SVR with GridSearch

tuned_parameters = [{'C': [1000, 3000, 10000], 
                     'kernel': ['linear', 'rbf']}
                   ]

#svr_tuned = GridSearchCV(SVR (C=1), param_grid = tuned_parameters, scoring = 'mean_squared_error') #default 3-fold cross-validation, score method of the estimator
svr_tuned_GS = GridSearchCV(SVR (C=1), param_grid = tuned_parameters, scoring = 'r2', n_jobs=-1) #default 3-fold cross-validation, score method of the estimator

svr_tuned_GS.fit(X_train, y_train)

print (svr_tuned_GS)
print ('\n' "Best parameter from grid search: " + str(svr_tuned_GS.best_params_) +'\n')


# Validation - SVR tuned 

svr_tuned_pred_GS = svr_tuned_GS.predict(X_test)

score_svr_tuned_GS = r2_score(y_test, svr_tuned_pred_GS)
rmse_svr_tuned_GS = sqrt(mean_squared_error(y_test, svr_tuned_pred_GS))

print("SVR Results\n")

print("Score SVR: %f" % score_svr)
print("Score SVR tuned GS: %f" % score_svr_tuned_GS)

print("\nRMSE SVR: %f" % rmse_svr)
print("RMSE SVR tuned GS: %f" % rmse_svr_tuned_GS)



# SVR tuned with RandomizesSearch
# may take a while!

# Parameters
param_dist = {  'C': sp_uniform (1000, 10000), 
                'kernel': ['linear']
             }

n_iter_search = 1

# MSE optimized
#SVR_tuned_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'mean_squared_error', n_iter=n_iter_search)

# R^2 optimized
SVR_tuned_RS = RandomizedSearchCV(SVR (C=1), param_distributions = param_dist, scoring = 'r2', n_iter=n_iter_search)

# Fit
SVR_tuned_RS.fit(X_train, y_train)

# Best score and corresponding parameters.
print('best CV score from grid search: {0:f}'.format(SVR_tuned_RS.best_score_))
print('corresponding parameters: {}'.format(SVR_tuned_RS.best_params_))

# Predict and score
predict = SVR_tuned_RS.predict(X_test)

score_svr_tuned_RS = r2_score(y_test, predict)
rmse_svr_tuned_RS = sqrt(mean_squared_error(y_test, predict))

predictions = pd.Series(predict, index = y_test.index.values)

plt.style.use('ggplot')
plt.figure(1)

plt.plot(y_test,'go', label='truth')
plt.plot(predictions,'bx', label='prediction')

plt.title('Number of bikes rented per day')
plt.xlabel('Days')
plt.xticks((np.arange(0,len(bike_Sharing_data),len(bike_Sharing_data)/24)), calendar.month_name[1:13]*2, rotation=45)

plt.ylabel('Number of bikes')

plt.legend(loc='best')

plt.show()

# Justification

print('score and rmse resultsof algorithms\n')

print("Score LR: %f" % score_lr)
print("Score DTR: %f" % score_dtr)
print("Score RFR: %f" % score_rfr)
print("Score GBR: %f" % score_gbr)
print("Score SVR: %f" % score_svr)

print("RMSE LR: %f" % rmse_lr)
print("RMSE DTR: %f" % rmse_dtr)
print("RMSE RFR: %f" % rmse_rfr)
print("RMSE GBR: %f" % rmse_gbr)
print("\nRMSE SVR: %f" % rmse_svr)


print('SVR Grid and Randomized Search Results\n')

print("Score SVR: %f" % score_svr)
print("Score SVR tuned GS: %f" % score_svr_tuned_GS)
print("Score SVR tuned RS: %f" % score_svr_tuned_RS)

print("\nRMSE SVR: %f" % rmse_svr)
print("RMSE SVR tuned GS: %f" % rmse_svr_tuned_GS)
print("RMSE SVR tuned RS: %f" % rmse_svr_tuned_RS)





