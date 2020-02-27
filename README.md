
# Spyder-Workshop

The main goal of this workshop is to explore some of Spyder's main functionalities for scientific programming. During this workshop we will work on data visualisation, analysis and prediction using Python libraries like Pandas, Matplotlib and Scikit-learn over a data set containing information about the weather history from 2006 to 2016.

## Project Set-Up
0. Clone https://github.com/juanis2112/Spyder-Workshop repository and in Spyder open this folder as a project under the 'Project' menu in the top toolbar. Then, open the file 'workshop.py' using the outline pane. 

```bash
git clone https://github.com/juanis2112/Spyder-Workshop
```

## Importing Libraries and Data
The first thing we need to do before starting our work, is importing the libraries necessary for our analysis and saving the data in a way that is easy to explore.
1. Import the libraries matplotlib and pandas.

2. Save the data from the csv file in a Pandas DataFrame using the command 

```python
weather_data = pd.read_csv('data/weatherHistory.csv')
```

## Exploring Data
Now that we have our data and our libraries ready, let's start by taking a look at the data that we have.

3. Open the weather_data variable in the Variable Explorer. 

4. Verify that the size of the data displayed in the Variable Explorer, corresponds to the result of the following command

```python
len(weather_data)
```

5. Print the first 3 rows of the DataFrame in the console using the command

```python
print(weather_data.head(3))
```

6. Now try printing the last 3 rows of the DataFrame.

## Visualisation
A useful tool for exploring the data that we are going to work with, is plotting it. This is easy to do, using our pandas library which we imported previously. 
The first thing we want to do before plotting our data is ordering the rows according to the date. Use the Variable Explorer to verify that our data is not ordered by default.

7. Use the following commands to create a new variable with our data ordered.

```python
weather_data['Formatted Date'] = pd.to_datetime(weather_data['Formatted Date'])

weather_data_ordered = weather_data.sort_values(by='Formatted Date')
```

8. Right click the old variable 'weather_data' to pop out the options menu and select 'Remove' to delete this variable. Now, we are going to work with our new variable 'weather_data_ordered'

Notice in the Variable Explorer that the column Index is now desorganized, with respect to the date. Use the following command 

```python
weather_data_ordered = weather_data_ordered.reset_index(drop=True)
```

to organize the Index column with respect to the "Formated Date".

We also see that there are some cualitative variables which make more difficult our analysis. For this reason, we want to stick to the columns that give us numerical information. Use the following command 

```python
weather_data_ordered = weather_data_ordered.drop(
    columns=['Summary', 'Precip Type', 'Loud Cover', 'Daily Summary'])
```

to drop categorical columns.

9. Plot the data of the Temperature (C) V.S the Date using the following command

```python
weather_data_ordered.plot(
    x='Formatted Date', y=['Temperature (C)'], color='red', figsize=(15, 8))
```
10. Open the Plots Pane to view your plot.
11. Now try plotting Temperature (C) V.S the Date using only the data from 2006 which corresponds to the first 8759 rows of the DataFrame.
12. Plot temperature and humidity V.S. the Date in the same plot, using the following command.

```python
weather_data_ordered.plot(subplots=True, x= 'Formatted Date', y= ['Temperature (C)', 'Humidity'],figsize=(15, 8))
```

13. Now try plotting different variables in the same plot for different years.

## Data Summarization and Aggregation
The previous plots contained a lot of data which make it difficult to understand the evolution of our variables through time. For this reason, one of the things that we can do is grouping the information we have by years and plot the value of the variables year by year. For this, we have created a function that creates a new column in the data frame containing the year, and then groups values by year, computing the average of the variables for each one. 

14. Import the function 

```python
from utils import (aggregate_by_year)
```

and use the following command

 ```python
 weather_data_by_year = aggregate_by_year(weather_data_ordered, 'Formatted Date')
```

to plot the values of the variables for each year.

15. Try writing a function in the utils.py file that gets the average of the weather data by month and plot it. 

## Data Analysis and Interpretation

Now, we want to evaluate the relationships between the variables in our data set. For this, we have written a function in the file 'utils.py' which should be in the same folder of your workshop. 

16. Import the function 

```python
from utils import (aggregate_by_year, plot_correlations) 
```

from this file, to be able to use it.

17. Plot the correlations between the variables using the command 

```python
plot_correlations(weather_data_ordered, size=15)
```

18. Open the plots pane to visualize the correlations plot.
19. Import the function 

```python
from utils import (aggregate_by_year, plot_correlations, plot_color_gradients) 
```

which is also in the utils.py file which will help you plot the colormap gradient to be able to interpret your correlations plot.
20. Plot the colormap gradient using the following commands.

```python
cmap_category, cmap_list = ('Plot gradients convention', ['viridis', ])

plot_color_gradients(cmap_category, cmap_list)
```

21. Calculate the correlations between the different variables in our data set usgin the following command 

```python
weather_correlations = weather_data_ordered.corr()
```

22. Open the variable `weather_correlations`in the Variable Explorer. 
23. Use the following command 

```python
weather_data_ordered['Temperature (C)'].corr(weather_data_ordered['Humidity'])
```

in the console to get the correlation between the Humidity and Temperature. Verify it has the same value in the correlations DataFrame.

24. Try calculating correlations between different variables and comparing them with the ones in the data frame.

## Data Modeling and Prediction
Finally, we want to use our data to construct a model that allows us predicting values for some of our variables. In our previous section we realized that humidity and temperature are two of the most correlated variables so we are going to use these two first. 

We are going to use scikit-learn which is a python library that contains tools to explore data and build different types of predictive models. We will use two functions for this task which need to be imported.

25. Use the following command to import the necessary libraries for our data modeling. 

```python
from sklearn.model_selection import train_test_split
from sklearn import linear_model
```

A classic way to make a predictive model is to subdivide the total set of data into two sets: training and test. The training data will help us to train our predictive model, while the test data will play the role of future observations and give us an idea is how good our prediction is.

26. Use the follwing command 

```python
X_train, X_test, Y_train, Y_test = train_test_split(weather_data_ordered['Humidity'], weather_data_ordered['Temperature (C)'],test_size=0.25)
```
to split your data.

We will use linear regression which is available in sklearn to make a linear model of our data.

27. Fit the weather data in a linear model using the following commands 

```python
regresion = linear_model.LinearRegression()

regresion.fit(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1))
```

28. Place the cursor over LinearRegression() and press `ctrl+i`to get the documentation of this funciton in the Help Pane.
29. Print the coefficients of our regression using 

```python
print(regresion.intercept_, regresion.coef_)
``` 

and save them in variables so we can use them with:

```python
beta_0 = regresion.intercept_[0]
beta_1 = regresion.coef_[0, 0]
``` 

Note that this means our model is a linear function `$$y = beta_0 + beta_1 \times x$$` where temperature is a function of temperature. 

30. Now we want to plot our predicted model and our test data, to see how good was our prediction. For this, use the following commands

```python
Y_predict = predicted_temperature(X_test, beta_0, beta_1)

plt.scatter(X_test, Y_test, c='red', label='observation', s=1)

plt.scatter(X_test, Y_predict, c='blue', label='model')

plt.xlabel('Humidity')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()
```

31. Using the coefficients found in our model, predict the temperature for a given level of humidity using the `predicted_temperature` function available in 'utils'.

Finally, we can numerically evaluate how good was our model predicted. For this we will use the `explained_variance_score` metric available in sklearn.metrics. This metric is calculated as 1-(Var(Y_real-Y_model)/Var(Y_real)) which means that the closer the value is to 1, the better our model. 

32. Use the following command 

```python
from sklearn.metrics import explained_variance_score
```

to import the function that evaluates how good is our model.

303. Calculate the explained variance score and print it using the following 

```python
ev = explained_variance_score(Y_test, Y_predict)
print(ev)
```










