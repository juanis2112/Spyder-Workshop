# -*- coding: utf-8 -*-
"""Workshop main flow."""


# %% [1] Import Packages and Data

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split

# Local imports
from utils import aggregate_by_year, plot_correlations, plot_color_gradients


# %% [2] Explore the Data

# Read data
weather_data = pd.read_csv('data/weatherHistory.csv')

# Print length of data
print(len(weather_data))

# Print first three rows of DataFrame
print(weather_data.head(3))

# TO DO: Print the last three rows of the DataFrame
print(weather_data.tail(3))


# %% [3] Visualization

# Order rows according to date
weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data_ordered = weather_data.sort_values(by='date')

# Reset index to restore its order
weather_data_ordered.reset_index(drop=True, inplace=True)

# Drop categorical columns
weather_data_ordered.drop(
    columns=['summary', 'precip_type', 'cloud_cover', 'daily_summary'],
    inplace=True)

# Plot temperature vs. date
weather_data_ordered.plot(
    x='date', y='temperature_c', color='red', figsize=(15, 8))

# TODO: Plot temperature vs date using only the data from 2006
weather_data_ordered.head(8759).plot(x='date', y=['temperature_c'], color='red')

# Plot temperature and humidity in the same plot
weather_data_ordered.plot(
    subplots=True, x='date', y=['temperature_c', 'humidity'],
    figsize=(15, 8))

# TODO: Plot different combinations of the variables, and for different years


# %% [4] Data summarization and aggregation

# Weather data by year
weather_data_by_year = aggregate_by_year(
    weather_data_ordered, 'date')

# TODO: Create and use a function to average the weather data by month


# %% [5] Data Analysis and Interpretation

# Plot correlations
plot_correlations(weather_data_ordered, size=15)

# Plot gradient colormaps
plot_color_gradients(
    cmap_category='Plot gradients convention', cmap_list=['viridis', ])

# Compute correlations
weather_correlations = weather_data_ordered.corr()
weather_data_ordered['temperature_c'].corr(
    weather_data_ordered['humidity'])

# TO DO: Get the correlation for different combinations of variables.
#       Contrast them with the weather_correlations dataframe


# %% [6] Data Modeling and Prediction

# Get data subsets for the model
x_train, x_test, y_train, y_test = train_test_split(
    weather_data_ordered['humidity'], weather_data_ordered['temperature_c'],
    test_size=0.25)

# Run regression
regression = linear_model.LinearRegression()
regression.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

# Print coefficients
print(regression.intercept_, regression.coef_)  # beta_0, beta_1


# %% [7] Predictive Model Testing and Evaluation

# Plot predicted model with test data
y_predict = regression.predict(x_test.values.reshape(-1, 1))
plt.scatter(x_test, y_test, c='red', label='Observation', s=1)
plt.scatter(x_test, y_predict, c='blue', label='Model')
plt.xlabel('humidity')
plt.ylabel('temperature_c')
plt.legend()
plt.show()

# TODO: Using the model, predict the temperature for a given level of humidity

# Evaluate model numerically
print(explained_variance_score(y_test, y_predict))
