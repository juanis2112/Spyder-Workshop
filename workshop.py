# -*- coding: utf-8 -*-
#
# Copyright © Spyder Project Contributors
# Licensed under the terms of the MIT License


# In[1] Importing Libraries and Data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2] Exploring Data

weather_data = pd.read_csv('data/weatherHistory.csv')
print(len(weather_data))
print(weather_data.head(3))

# TODO: Print the last 3 rows of the DataFrame 



# In[3] Visualisation

weather_data['Formatted Date'] = pd.to_datetime(weather_data['Formatted Date'])
weather_data_ordered = weather_data.sort_values(by='Formatted Date')

weather_data_ordered = weather_data_ordered.reset_index(drop=True)
weather_data_ordered.plot(x='Formatted Date', y=['Temperature (C)'], color='red')

# TODO: Plot Temperature (C) V.S the Date using only the data from 2006



#-----------------------------------------------------------------------------

weather_data_ordered.plot(subplots=True, x= 'Formatted Date', y= ['Temperature (C)', 'Humidity'])

# TODO: Plot different combinations of the variables, for different years



#-----------------------------------------------------------------------------

# In[4] 






