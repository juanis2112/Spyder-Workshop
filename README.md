# Spyder-Workshop

This workshop explores some of the Spyder Python IDE's core functionality for scientific programming.
We will work on data visualization, analysis and prediction using packages like Pandas, Matplotlib and Scikit-learn over a dataset with historical weather observations from 2006 to 2016.


## Prerequisites

To use this tutorial, you will need to have [Spyder](https://www.spyder-ide.org) installed in a Python environment that contains at least the Numpy, Matplotlib, Pandas and Scikit-Learn packages.
We recommend you [download the Anaconda Python distribution](https://www.anaconda.com/products/individual) which contains all these packages and more all in one easy-to-install place.

You will also need to [download Git](https://git-scm.com/downloads), the popular version control tool, to work with the contents of this repository.


## Project Set-Up

0. From the command line, clone the [Spyder-Workshop repository](https://github.com/juanis2112/Spyder-Workshop) with git:

```bash
git clone https://github.com/juanis2112/Spyder-Workshop
```

Then, launch Spyder (via the start menu shortcut on Windows, or from Anaconda Navigator or `spyder` on the command line with Mac or Linux).
Open the resulting folder in Spyder as a project by clicking `Open Project` in the `Project` menu, and navigating to the `Spyder-Workshop` directory you cloned.
Finally, open the file `workshop.py` by double-clicking it in the `Project Explorer` pane to the left of the Spyder main window.


## Import Packages and Data

Before starting our work, we need to import the packages necessary for our analysis and load the data in a way that it's easy to explore.

1. Import matplotlib and pandas:

```python
import matplotlib.pyplot as plt
import pandas as pd
```

2. Load the data from the CSV file to Pandas DataFrame:

```python
weather_data = pd.read_csv('data/weatherHistory.csv')
```


## Explore the Data

Now that we have our files and packages ready, let's start by taking a look at the data that we have.

3. Open the `weather_data` variable in Spyder's Variable Explorer pane by double-clicking its name.
The Variable Explorer is located in the top-right of the Spyder main window; you may need to click its tab to make it visible.

4. Verify that the `Size` of `weather_data` in the Variable Explorer corresponds to the length given by running the `len()` function on your DataFrame:

```python
print(len(weather_data))
```

5. Then, print the first three rows of the DataFrame by using its `head()` method:

```python
print(weather_data.head(3))
```

6. Now, try printing the last three rows of the DataFrame.


## Visualization

Plotting is useful tool for exploring the data that we are going to work with.
This is easy to do using our Pandas DataFrame.

Before plotting our data, we can order the rows according to the date.
Use the Variable Explorer to verify that our data is not ordered by default.

7. First, parse the date and create a new DataFrame with our data ordered by it:

```python
weather_data['Formatted Date'] = pd.to_datetime(weather_data['Formatted Date'])
weather_data_ordered = weather_data.sort_values(by='Formatted Date')
```

8. In the Variable Explorer, right-click the old DataFrame `weather_data` to view the context menu and select `Remove` to delete this object.
Now, we are going to work with our new variable `weather_data_ordered`.

Notice in the Variable Explorer that the DataFrame's index (the `Index` column on the left) is now out of order.
Reset the index to be in order again:

```python
weather_data_ordered = weather_data_ordered.reset_index(drop=True)
```

We also see that there are some qualitative variables, which can make our analysis more difficult.
For this reason, we want to filter out these columns, and stick to the ones that give us numerical information:

```python
weather_data_ordered = weather_data_ordered.drop(
    columns=['Summary', 'Precip Type', 'Loud Cover', 'Daily Summary'])
```

9. Plot the `Temperature (C)` column versus the `Date` column to see how temperature changes over time:

```python
weather_data_ordered.plot(
    x='Formatted Date', y=['Temperature (C)'], color='red', figsize=(15, 8))
```

10. Switch to Spyder's Plots pane, in the same top-right section of the interface as the Variable Explorer, to view your figure.

11. Now, try plotting the same columns using only the data from 2006.

12. Plot both the `Temperature (C)` and `Humidity` columns versus the `Date` column to examine how both variables change over the years:

```python
weather_data_ordered.plot(subplots=True, x= 'Formatted Date', y= ['Temperature (C)', 'Humidity'],figsize=(15, 8))
```

13. Explore your data!
Try plotting different variables in the same plot for different years.


## Data Summarization and Aggregation

The previous plots contained a lot of data, which make it difficult to understand the evolution of our variables through time.
For this reason, we can group the data by year and plot just the yearly values.
We have written a function for you in the `utils.py` file that creates a new column in the DataFrame containing the year, and then groups values by year, computing the average of the variables for each one.

14. Import the function from the `utils` module, so you can use it in your script:

```python
from utils import (aggregate_by_year)
```

15. Then, use it to aggregate by year and plot the data:

 ```python
 weather_data_by_year = aggregate_by_year(weather_data_ordered, 'Formatted Date')
```

16. Try writing your own function that averages the weather data by month and plots it.


## Data Analysis and Interpretation

Now, we want to evaluate the relationships between the variables in our data set.
For this, we have written another function in `utils.py`.

17. First, import the new function:

```python
from utils import (aggregate_by_year, plot_correlations)
```

18. Now, use it to plot the correlations between the variables:

```python
plot_correlations(weather_data_ordered, size=15)
```

Like before, open the Plots pane to view the correlations plot.

19. Now, import the `plot_color_gradients()` function, which will help you plot the colormap gradient to be able to interpret your correlation plot:

```python
from utils import (aggregate_by_year, plot_correlations, plot_color_gradients)
```

20. Plot the colormap gradient using the function you imported:

```python
cmap_category, cmap_list = ('Plot gradients convention', ['viridis', ])
plot_color_gradients(cmap_category, cmap_list)
```

21. Now, calculate the (Pearson) correlations between the different variables in our data set:

```python
weather_correlations = weather_data_ordered.corr()
```

22. Open the object  `weather_correlations` in the Variable Explorer to see the results.

23. You can also get just the correlation between two particular variables, such as temperature and humidity:

```python
weather_data_ordered['Temperature (C)'].corr(weather_data_ordered['Humidity'])
```

Verify it has the same value as in the `weather_correlations` DataFrame.

24. Try calculating correlations between other variables and comparing them with the ones in the DataFrame.


## Data Modeling and Prediction

Finally, we want to use our data to construct a model that allows us to predict future values for some of our variables.
From the previous section, you may have noticed that temperature and humidity are two of the most correlated variables, so we are going to look at those first.

We are going to use Scikit-Learn, which is a Python package that contains tools to explore data and build different types of machine learning models.

25. We will use two functions for this task which need to be imported first:

```python
from sklearn.model_selection import train_test_split
from sklearn import linear_model
```

26. A classic way to make a predictive model is to subdivide the total set of data into two: training and test.
The training data will help us to train our model, while the test data will play the role of future observations and give us an idea of how good our prediction is.
Scikit-Learn contains a built-in function to split your data:

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    weather_data_ordered['Humidity'], weather_data_ordered['Temperature (C)'],
    test_size=0.25)
```

27. We will use linear regression to make a linear model of our data.
Create the model, then fit it with the weather data:

```python
regresion = linear_model.LinearRegression()
regresion.fit(X_train.values.reshape(-1, 1), Y_train.values.reshape(-1, 1))
```

28. To get the function's documentation in Spyder's Help Pane, place the text cursor over `LinearRegression()` and press the `Inspect` shortcut (`Ctrl+I` by default on Windows/Linux, or `Cmd-I` on macOS) .

29. Finally, print the coefficients of our regression:

```python
print(regresion.intercept_, regresion.coef_)
```

and save them in variables so we can use them later:

```python
beta_0 = regresion.intercept_[0]
beta_1 = regresion.coef_[0, 0]
```

Note that this means our model is a linear function `$$y = beta_0 + beta_1 \times x$$`, where future temperature is a function of past temperature.


## Predictive Model Testing and Evaluation

30. First, we want to plot our model predictions vs. our test data to see how accurate our prediction was:

```python
Y_predict = predicted_temperature(X_test, beta_0, beta_1)
plt.scatter(X_test, Y_test, c='red', label='observation', s=1)
plt.scatter(X_test, Y_predict, c='blue', label='model')

plt.xlabel('Humidity')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()
```

31. Using the coefficients found in our model, predict the temperature for a given level of humidity using the `predicted_temperature` function available in 'utils.py'.

32. Finally, we want to evaluate our model performance.
For this we will use the `explained_variance_score` metric available in `sklearn.metrics`.
This metric is calculated as `$$1-(Var(Y_real-Y_model)/Var(Y_real))$$`, which means that the closer the value is to 1, the better our model.
We first need to import the appropriate function:

```python
from sklearn.metrics import explained_variance_score
```

33. Finally, calculate the explained variance score and print it:

```python
ev = explained_variance_score(Y_test, Y_predict)
print(ev)
```
