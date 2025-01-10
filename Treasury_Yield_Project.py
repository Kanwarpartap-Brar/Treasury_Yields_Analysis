#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: shawnbrar
"""
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



#PART 1 (Question Formulation)
#“Are changes in treasury yields leading indicators of changes in mortgage rates, or do they follow a lagged relationship?”


#PART 2 (Data Collection)

#read treasury yield data, 15-year mortgage rate data, and 30-year mortgage rate data

treasury_yield_data = pd.read_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/Final_Project/Treasury_Yield_Data.csv')
#print(treasury_yield_data.head())

#rename DGS10 column to yield
treasury_yield_data.rename(columns = {'DGS10':'Yield'},inplace = True)

Mortgage_Data_15 = pd.read_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/Final_Project/15_Year_Mortgage_Data.csv')

Mortgage_Data_30 = pd.read_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/Final_Project/30_Year_Mortgage_Data.csv')



#check to see if any null values in the data
#print('\n')
#print('Tres Data: ')
#print(treasury_yield_data.isnull().any())
#print(treasury_yield_data.info())
#checking for non-numeric values b/c Yield is an object not float64
#print(treasury_yield_data['Yield'].unique()) 


#convert date to datetime instead of object 

treasury_yield_data['DATE'] = pd.to_datetime(treasury_yield_data['DATE'])
Mortgage_Data_15['DATE'] = pd.to_datetime(Mortgage_Data_15['DATE'])
Mortgage_Data_30['DATE'] = pd.to_datetime(Mortgage_Data_30['DATE'])


#acciental '.' instead of a value so I will replace convert column to numeric and then use linear interpolation to fill missing value
treasury_yield_data['Yield'] = treasury_yield_data['Yield'].replace('.', np.nan) 
treasury_yield_data['Yield'] = pd.to_numeric(treasury_yield_data['Yield'], errors='coerce')
treasury_yield_data['Yield'] = treasury_yield_data['Yield'].interpolate(method='linear')


print('\n')
print('15-Year Data: ')
print(Mortgage_Data_15.isnull().any())
print(Mortgage_Data_15.info())

print('\n')
print('30-Year Data: ')
print(Mortgage_Data_30.isnull().any())
print(Mortgage_Data_30.info())

print('\n')
print('Tres Info: ')
print(treasury_yield_data.isnull().any())
print(treasury_yield_data.info()) #Yield has been converted to a numeric datatype


print(treasury_yield_data.head())
print(Mortgage_Data_15.head())
print(Mortgage_Data_30.head())



#merge 3 datasets using inner join so they all have the same dates (weekly dates)

fp_merged_data = pd.merge(treasury_yield_data, Mortgage_Data_15, on = 'DATE', how = 'inner')
fp_merged_data = pd.merge(fp_merged_data, Mortgage_Data_30, on = 'DATE', how = 'inner')

#checking to see if data is clean
print('\n')
print('Merged Data Info: ')
print(fp_merged_data.isnull().any())
print(fp_merged_data.shape)
print(fp_merged_data.info())

#saving my merged dataset as a csv file
#fp_merged_data.to_csv('/Users/shawnbrar/Desktop/Hofstra_Classes/DS_221/Final_Project/fp_merged_data.csv')


#PART 3 (EDA)

##1. summary statistics

summary_statistics = fp_merged_data.describe()
#gets mean, median, sd, min, max, quartiles 
#print(summary_statistics)

numeric_columns = fp_merged_data.select_dtypes(include = 'float64') #selecting all columns except 'Date'
var = numeric_columns.var()  #getting variance of columns
summary_statistics.loc['var'] = var
summary_statistics = summary_statistics.drop(columns = 'DATE') #drop 'Date' as it is unnecessary 
print("\n")
print("Summmary Statistics: ")
print(summary_statistics) 

##2. Data Visualization

#print(fp_merged_data.head())

#area chart using plotly 

fp_long = fp_merged_data.melt(id_vars = 'DATE', value_vars = ['Yield', 'MORTGAGE15US', 'MORTGAGE30US'], var_name='Rate Type', value_name='Rate (%)')


fig = px.area(fp_long, x = 'DATE', y = 'Rate (%)', color = 'Rate Type', line_group = 'Rate Type',
labels = { 'Rate (%)': 'Interest Rate (%)', 'variable':'Rate Type'},
title = 'Interest Rates Over Time')

fig.update_layout(width = 1000, height = 500)

fig.show()

#box plot for each variable

#Yield box plot 

fig_yield = px.box(fp_merged_data, y = 'Yield', points = 'all',
                   labels = {'Yield': 'Treasury Yield(%)' },
                   title = 'Distribution of Treasury Yield')

fig.update_layout(width = 1000, height = 500)
fig_yield.show() 

#MORTGAGE15US box plot

fig = px.box(fp_long, x = 'Rate Type', y = 'Rate (%)', color = 'Rate Type', points = 'all', 
             title = 'Rate Distribution')
fig.update_layout(width = 1000, height = 500)
fig.show()

##3. DATA DISTRIBUTION

#histogram for yield 

plt.hist(fp_merged_data['Yield'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('Treasury Yield (%)')
plt.ylabel('Count')
plt.title('Treasury Yield Distribution')
plt.show()

#histogram for MORTGAGE15US 

plt.hist(fp_merged_data['MORTGAGE15US'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Count')
plt.title('15 Year Mortgage Rate Distribution')
plt.show()

#histogram for MORTGAGE30US

plt.hist(fp_merged_data['MORTGAGE30US'], edgecolor = 'black', color = 'royalblue')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Count')
plt.title('30 Year Mortgage Rate Distribution')
plt.show()

#measure value of skewness (positive = right, negative = left skew)

skewness_Yield = fp_merged_data['Yield'].skew()
print('Skewness of Yield:', skewness_Yield)

skewness_M15 = fp_merged_data['MORTGAGE15US'].skew()
print('\n')
print('Skewness of MORTGAGE15US:', skewness_M15)

skewness_M30 = fp_merged_data['MORTGAGE30US'].skew()
print('\n')
print('Skewness of MORTGAGE30US:', skewness_M30)

##4. CORRELATION ANALYSIS



fp_merged_data['Yield'] = pd.to_numeric(fp_merged_data['Yield'], errors='coerce')

fig = go.Figure()

#15 year mortgage rate scatter plot

fig.add_trace(go.Scatter(x = fp_merged_data['Yield'], y = fp_merged_data['MORTGAGE15US'], mode = 'markers',
                         name = '15 Year Mortgage Rate', marker = dict(color = 'purple', symbol = 'triangle-up')))

#30 year mortgage rate scatter plot

fig.add_trace(go.Scatter(x = fp_merged_data['Yield'], y = fp_merged_data['MORTGAGE30US'], mode = 'markers',
                         name = '30 Year Mortgage Rate', marker = dict(color = 'gold', symbol = 'square')))

fig.update_layout(title = 'Correlation Analysis Between Treasury Yields & Mortgage Rates', xaxis_title = 'Treasury Yield (%)',
                  yaxis_title = 'Mortgage Rate (%)', legend_title = 'Mortgage Type')

fig.show()

##5. HYPOTHESIS TESTING 

#Is there a time lag between changes in treasury yields mortgage rates?

#find differneces in changes in interest rates

fp_merged_data['Yield_changes'] = fp_merged_data['Yield'].diff()  #changes in Yield

fp_merged_data['M15_changes'] = fp_merged_data['MORTGAGE15US'].diff() #changes in 15 year mortgage data

fp_merged_data['M30_changes'] = fp_merged_data['MORTGAGE30US'].diff() #changes in 30 year mortgage data

fp_merged_data = fp_merged_data.dropna() #drops first row bc you cant difference the first row so it will not have anumber and will be null

#run ordinary least square regression

X = sm.add_constant(fp_merged_data['Yield_changes'])  #constant term 

#dependednt variables whos relationship to yield changes is being observed
mort_15_change = fp_merged_data['M15_changes']
mort_30_change = fp_merged_data['M30_changes']

#OLS for 15year rates
OLS_M15 = sm.OLS(mort_15_change, X).fit()

#OLS for 30 year rates
OLS_M30 = sm.OLS(mort_30_change, X).fit()

change_on_15yr = OLS_M15.params['Yield_changes']
change_on_30yr = OLS_M30.params['Yield_changes']

#conduct ttest

statistic, p_value = ttest_ind(fp_merged_data['M15_changes'], fp_merged_data['M30_changes'], equal_var=False)

print("\n")
print('HYPOTHESIS TESTING')
print("t_stat: ", statistic, "p_value: ", p_value) 

alpha = .05
if p_value < .05:
    print("Reject the null hypothesis, the result is statistically significant at the 5% significance level")
else:
    print("Failed to reject the null hypothesis")


#4. MACHINE LEARNING


#predict 30 year mortgage rates by predicting yields
#predict the treasury yield rate by predicting 30 year mortgage rates


#predict 15 year mortgage rates 
#Using a forest regressor model

X = fp_merged_data[['Yield', 'MORTGAGE30US']] #feature variables (dependent)
y = fp_merged_data['MORTGAGE15US'] #target varialbe (independent)

#split data so 20% is for testing and 80% is for training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

random_forest_M15 = RandomForestRegressor(n_estimators= 100, random_state = 42) #100 estimators for a better prediction

#train model

random_forest_M15.fit(X_train, y_train)

#predict
y_pred_M15 = random_forest_M15.predict(X_test)

print('\n')
print('Predicted 15-year Mortgage Rates: ')
print(y_pred_M15)


#model evaluation

mse_random_forest_M15 = mean_squared_error(y_test, y_pred_M15) #get mse 
rmse_random_forest_M15 = np.sqrt(mse_random_forest_M15) #get rmse
r2_random_forest_M15 = r2_score(y_test, y_pred_M15) #get r^2



print('\n')
print("Mean Squared Error Of Random Forest Regressor Model (M15): ", mse_random_forest_M15)
print('RMSE Of Random Forest Regressor Model (M15): ', rmse_random_forest_M15)
print("R2 Score of Random Forest Regressor Model (M15): ", r2_random_forest_M15 )

#visualize the model and its predictions

dates = pd.date_range(start='2020-01-01', periods=len(y_test), freq='ME') #monthly range of dates

actual_M15 = y_test #actual points
predicted_M15 = y_pred_M15 #model predicted rates

plt.figure(figsize=(10, 5))
plt.plot(dates, actual_M15, label = 'Actual 15-Year Mortgage Rates', color = 'gold', marker = 'o')
plt.plot(dates, predicted_M15, label = 'Predicted 15-Year Mortgage Rates', color = 'purple', marker = 'x')
plt.title('Predicted VS Actual 15-Year Mortgage Rates')
plt.xlabel('Date')
plt.ylabel('15-Year Mortgage Rate (%)')
plt.legend()
plt.grid(True)
plt.show()


#predict 30 year mortgage rates 
#Using a forest regressor model

X_M30 = fp_merged_data[['Yield', 'MORTGAGE15US']] #feature variables (dependent)
y_M30 = fp_merged_data['MORTGAGE30US'] #target varialbe (independent)

#split data so 20% is for testing and 80% is for training
X_M30_train, X_M30_test, y_M30_train, y_M30_test = train_test_split(X_M30, y_M30, test_size = 0.2, random_state = 42)

random_forest_M30 = RandomForestRegressor(n_estimators= 100, random_state = 42) #100 estimators for a better prediction


#train model
random_forest_M30.fit(X_M30_train, y_M30_train)

#predict model
y_pred_M30 = random_forest_M30.predict(X_M30_test)


print('\n')
print('Predicted 30-year Mortgage Rates: ')
print(y_pred_M30)


#model evaluation

mse_random_forest_M30 = mean_squared_error(y_M30_test, y_pred_M30) #get mse 
rmse_random_forest_M30 = np.sqrt(mse_random_forest_M30) #get rmse
r2_random_forest_M30 = r2_score(y_M30_test, y_pred_M30) #get r^2

print('\n')
print("Mean Squared Error Of Random Forest Regressor Model (M30): ", mse_random_forest_M30)
print('RMSE Of Random Forest Regressor Model (M30): ', rmse_random_forest_M30)
print("R2 Score of Random Forest Regressor Model (M30): ", r2_random_forest_M30 )

#visualize the model and its predictions

dates_M30 = pd.date_range(start='2020-01-01', periods=len(y_M30_test), freq='ME') #monthly range of dates

actual_M30 = y_M30_test #actual points
predicted_M30 = y_pred_M30 #model predicted rates

plt.figure(figsize=(10, 5))
plt.plot(dates_M30, actual_M30, label = 'Actual 30-Year Mortgage Rates', color = 'gold', marker = 'o')
plt.plot(dates_M30, predicted_M30, label = 'Predicted 30-Year Mortgage Rates', color = 'purple', marker = 'x')
plt.title('Predicted VS Actual 30-Year Mortgage Rates')
plt.xlabel('Date')
plt.ylabel('30-Year Mortgage Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

#predict treasury yields using mortgage rates
#using a forest regressor model

X_Yield = fp_merged_data[['MORTGAGE30US', 'MORTGAGE15US']] #feature variables (dependent)
y_Yield = fp_merged_data['Yield'] #target varialbe (independent)

#split data so 20% is for testing and 80% is for training
X_Yield_train, X_Yield_test, y_Yield_train, y_Yield_test = train_test_split(X_Yield, y_Yield, test_size = 0.2, random_state = 42)

random_forest_Yield = RandomForestRegressor(n_estimators= 100, random_state = 42) #100 estimators for a better prediction

#train model
random_forest_Yield.fit(X_Yield_train, y_Yield_train)

#predict model
y_pred_Yield = random_forest_Yield.predict(X_Yield_test)

print('\n')
print('Predicted Treasury Yield Rates: ')
print(y_pred_Yield)

#model evaluation

mse_random_forest_Yield = mean_squared_error(y_Yield_test, y_pred_Yield) #get mse 
rmse_random_forest_Yield = np.sqrt(mse_random_forest_Yield) #get rmse
r2_random_forest_Yield = r2_score(y_Yield_test, y_pred_Yield) #get r^2

print('\n')
print("Mean Squared Error Of Random Forest Regressor Model (Treasury Yield Rate): ", mse_random_forest_Yield)
print('RMSE Of Random Forest Regressor Model (Treasury Yield Rate): ', rmse_random_forest_Yield)
print("R2 Score of Random Forest Regressor Model (Treasury Yield Rate): ", r2_random_forest_Yield)

#visualize the model and its predictions

dates_Yield = pd.date_range(start='2020-01-01', periods=len(y_Yield_test), freq='ME') #monthly range of dates

actual_Yield = y_Yield_test #actual points
predicted_Yield = y_pred_Yield #model predicted rates

plt.figure(figsize=(10, 5))
plt.plot(dates_Yield, actual_Yield, label = 'Actual Treasury Yield Rates', color = 'gold', marker = 'o')
plt.plot(dates_Yield, predicted_Yield, label = 'Predicted Treasury Yield Rates', color = 'purple', marker = 'x')
plt.title('Predicted VS Actual Treasury Yield Rates')
plt.xlabel('Date')
plt.ylabel('Treasury Yield Rate (%)')
plt.legend()
plt.grid(True)
plt.show()














