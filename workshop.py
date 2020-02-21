# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License
"""Workshop main flow."""


# In[1] Importing Libraries and Data

# Third-party imports
import matplotlib  # Needed for the use of pandas DataFrame.plot
import pandas as pd

# Local imports
from utils import plot_correlations

# In[2] Exploring Data

weather_data = pd.read_csv('data/weatherHistory.csv')
print(len(weather_data))
print(weather_data.head(3))

# Drop categorical columns
weather_data = weather_data.drop(
    columns=['Summary', 'Precip Type', 'Loud Cover', 'Daily Summary'])


# TODO: Print the last 3 rows of the DataFrame


# In[3] Visualisation

weather_data['Formatted Date'] = pd.to_datetime(
    weather_data['Formatted Date'])
weather_data_ordered = weather_data.sort_values(by='Formatted Date')

weather_data_ordered = weather_data_ordered.reset_index(drop=True)
weather_data_ordered.plot(
    x='Formatted Date', y=['Temperature (C)'], color='red', figsize=(15, 8))

# TODO: Plot Temperature (C) V.S the Date using only the data from 2006


# -----------------------------------------------------------------------------
weather_data_ordered.plot(
    subplots=True, x='Formatted Date', y=['Temperature (C)', 'Humidity'],
    figsize=(15, 8))

# TODO: Plot different combinations of the variables, for different years


# -----------------------------------------------------------------------------


# In[4] Correlations
plot_correlations(weather_data_ordered, size=15)

weather_corr_temp_humidity = weather_data_ordered['Temperature (C)'].corr(
    weather_data_ordered['Humidity'])

# TODO: Get the correlation between
