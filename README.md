# Spyder Workshop

The main goal of this workshop is to explore some of the Spyder IDE's core functionality for scientific programming.
We will work on data visualization, analysis and prediction using Python packages like Pandas, Matplotlib and Scikit-learn over a dataset of historical weather observations from 2006 to 2016.


## Prerequisites

To follow along with this workshop, you will need to have [Spyder](https://www.spyder-ide.org) installed in a Python environment that contains at least the Numpy, Matplotlib, Pandas and Scikit-Learn packages.
We recommend you [download the Anaconda Python distribution](https://www.anaconda.com/products/individual) which contains all these packages and more all in one easy-to-install place.

You will also need to [download Git](https://git-scm.com/downloads), the popular version control tool, to work with the contents of this repository.


## Project Set-Up

0. From the command line, clone the [Spyder-Workshop repository](https://github.com/juanis2112/Spyder-Workshop) with git:

```bash
git clone https://github.com/juanis2112/Spyder-Workshop
```

Then, launch Spyder (via the start menu shortcut on Windows, or from Anaconda Navigator or `spyder` on the command line with Mac or Linux).
Open the cloned folder in Spyder as a project by clicking `Open Project` in the `Project` menu, and navigating to the `Spyder-Workshop` directory you cloned.
Finally, open the file `workshop.py` by double-clicking it in the `Project Explorer` pane to the left of the Spyder main window.


## Import Packages and Data

The first thing we need to do before starting our work is import the packages necessary for our analysis, and load the data in a way that it is easy to explore.

1. Import the packages Matplotlib and Pandas:

```python
import matplotlib.pyplot as plt
import pandas as pd
```

2. Load the data from the CSV file to a Pandas DataFrame:

```python
weather_data = pd.read_csv('data/weatherHistory.csv')
```


## Explore the Data

Now that we have our data and packages ready, let's start by taking a look at the data that we have.

3. Open the `weather_data` variable in Spyder's Variable Explorer pane by double-clicking its name.
The Variable Explorer is located in the top-right of the Spyder main window; you may need to click its tab to make it visible.

4. Verify that the `Size` of `weather_data` in the Variable Explorer corresponds to the result of the `len()` function on your DataFrame:

```python
print(len(weather_data))
```

5. Print the first three rows of the DataFrame to the IPython Console:

```python
print(weather_data.head(3))
```

6. Now, try printing the last three rows of the DataFrame.


## Visualization

A useful tool for exploring the data that we are going to work with is plotting it.
This is easy to do using the pandas library, which we imported previously.

The first thing we want to do before plotting our data is ordering the rows according to the date.
Use the Variable Explorer to verify that our data is not ordered by default.

7. Parse the date and create a new DataFrame with our data ordered by it:

```python
weather_data['date'] = pd.to_datetime(weather_data['date'])
weather_data_ordered = weather_data.sort_values(by='date')
```

8. In the Variable Explorer, right-click the old DataFrame `weather_data` to pop out the context menu and select `Remove` to delete it.
Now, we are going to work with our new variable `weather_data_ordered`.

Notice in the Variable Explorer that the DataFrame's index (the `Index` column on the left) is not in the order of the date.
Reset the index so its order matches that of `date`:

```python
weather_data_ordered.reset_index(drop=True, inplace=True)
```

We also see that there are some qualitative variables, which can make our analysis more difficult.
For this reason, we want to stick to the columns that give us numerical information and drop the categorical ones:

```python
weather_data_ordered.drop(
    columns=['summary', 'precip_type', 'cloud_cover', 'daily_summary'],
    inplace=True)
```

9. Plot the temperature versus the date to see how temperature changes over time:

```python
weather_data_ordered.plot(
    x='date', y='temperature_c', color='red', figsize=(15, 8))
```

10. Open the Plots pane, in the same top-right section of the interface as the Variable Explorer, to view your figure.

11. Now, try plotting the temperature versus the date using only the data from 2006.

12. Plot temperature and humidity versus the date in the same plot to examine how both variables change over time:

```python
weather_data_ordered.plot(
    subplots=True, x='date', y=['temperature_c', 'humidity'],
    figsize=(15, 8))
```

13. Now, try plotting different variables in the same plot for different years.


## Data Summarization and Aggregation

The previous plots contained a lot of data, which make it difficult to understand the evolution of our variables through time.
For this reason, one of the things that we can do is group the information we have by year and plot the yearly values.
To do this, we have written a function in the `utils.py` file, in the same folder as your workshop, that creates a new column in the DataFrame containing the year, and then groups values by year, computing the average of the variables for each one.

14. Import the function from the `utils` module, so you can use it in your script:

```python
from utils import aggregate_by_year
```

15. Use it to aggregate by year and plot the data:

```python
weather_data_by_year = aggregate_by_year(
    weather_data_ordered, 'date')
```

16. Try writing a function in the `utils.py` file that gets the averages of the weather data by month and plots them.


## Data Analysis and Interpretation

Now, we want to evaluate the relationships between the variables in our data set.
For this, we have written another function in `utils.py`.

17. Import the new function:

```python
from utils import aggregate_by_year, plot_correlations
```

18. Plot the correlations between the variables:

```python
plot_correlations(weather_data_ordered, size=15)
```

Like before, open the Plots pane to view the correlations plot.

19. Import the `plot_color_gradients()` function, which will help you plot the colormap gradient to be able to interpret your correlation plot:

```python
from utils import aggregate_by_year, plot_correlations, plot_color_gradients
```

20. Plot the colormap gradient using the function you imported:

```python
plot_color_gradients(
    cmap_category='Plot gradients convention', cmap_list=['viridis', ])
```

21. Calculate the Pearson correlations between the different variables in our data set:

```python
weather_correlations = weather_data_ordered.corr()
```

22. Open the variable `weather_correlations` in the Variable Explorer to see the results.

23. Print the correlation between humidity and temperature in the IPython Console:

```python
weather_data_ordered['temperature_c'].corr(weather_data_ordered['humidity'])
```

Verify it has the same value as in the `weather_correlations` DataFrame.

24. Try calculating correlations between different variables and comparing them with the ones in the DataFrame.


## Data Modeling and Prediction

Finally, we want to use our data to construct a model that allows us to predict values for some of our variables.
In our previous section, we realized that humidity and temperature are two of the most correlated variables, so we are going to use these two first.

We are going to use Scikit-Learn, which is a Python package that contains tools to explore data and build different types of predictive models.

25. Import the two necessary objects for our data modeling:

```python
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```

26. A classic way to make a predictive model is to subdivide the data into two sets: training and test.
The training data will help us to fit our predictive model, while the test data will play the role of future observations and give us an idea of how good our predictions are.

Scikit-Learn contains a built-in function to split your data:

```python
x_train, x_test, y_train, y_test = train_test_split(
    weather_data_ordered['humidity'], weather_data_ordered['temperature_c'],
    test_size=0.25)
```

27. We will use linear regression in Scikit-Learn to make a linear model of our data.
Create the model, then fit it with the weather data:

```python
regression = linear_model.LinearRegression()
regression.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
```

28. Place the text cursor over `LinearRegression()` and press the `Inspect` shortcut (`Ctrl+I` by default on Windows/Linux, or `Cmd-I` on macOS) to get the documentation of this function in the Help Pane.

29. Print the coefficients of our regression:

```python
print(regression.intercept_, regression.coef_)  # beta_0, beta_1
```

Note that this means our model is a linear function `$$y = beta_0 + beta_1 \times x$$`, where temperature is a function of humidity.


## Predictive Model Testing and Evaluation

30. Now, we want to plot our model predictions versus our test data, to see how good our predictions were:

```python
y_predict = regression.predict(x_test.values.reshape(-1, 1))
plt.scatter(x_test, y_test, c='red', label='Observation', s=1)
plt.scatter(x_test, y_predict, c='blue', label='model')
plt.xlabel('humidity')
plt.ylabel('temperature_c')
plt.legend()
plt.show()
```

31. Using the `.predict()` method of our model, predict the temperature for a given level of humidity.

32. Finally, we can numerically evaluate how good our model predictions were.
For this, we will use `explained_variance_score` available in `sklearn.metrics`.
This metric is calculated as `$$1-(Var(Y_real-Y_model)/Var(Y_real))$$`, which means that the closer the value is to 1, the better our model.

We need to import the function that evaluates our model:

```python
from sklearn.metrics import explained_variance_score
```

33. Calculate the explained variance score and print it:

```python
print(explained_variance_score(y_test, y_predict))
```
