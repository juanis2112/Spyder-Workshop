# Spyder-Workshop
Spyder Workshop 

## Importing Libraries and Data
1. Import the libraries matplotlib, numpy and pandas.
2. Save the data from the csv file in a Pandas DataFrame using the command 
`weather_data = pd.read_csv('data/weatherHistory.csv')`.

## Exploring Data
3. Open the weather_data variable in the Variable Explorer. 
4. Verify that the size of the data displayed in the Variable Explorer, corresponds to the result of the following command
`len(weather_data)`
5. Print the first 3 rows of the DataFrame in the console using the command
`print(weather_data.head(3))`
6. Now try printing the last 3 rows of the DataFrame.

## Visualisation
The first thing we want to do before plotting our data is ordering the rows according to the date. Use the Variable Explorer to see that our Data is not ordered by default.

7. Use the following commands to create a new variable with our data ordered.
`weather_data['Formatted Date'] = pd.to_datetime(weather_data['Formatted Date'])`
`weather_data_ordered = weather_data.sort_values(by='Formatted Date')`
8. Right click the old variable 'weather_data' to pop out the options menu and select 'Remove' to delete this variable. Now, we are going to work with our new variable 'weather_data_ordered'
Notice in the Variable Explorer that the column Index is now desorganized, with respect to the Date. For this, we want to create a new column 'Index' with out values ordered.
9. Delete the column 
9. Plot the data of the Temperature (C) V.S the Date using the following command
`weather_data_ordered.plot(x='Formatted Date', y=['Temperature (C)'], color='red')`
10. Open the Plots Pane to view your plot.
11. Now try plotting Temperature (C) V.S the Date using only the data from 2006 which corresponds to the first 8759 rows of the DataFrame.
12. Plot temperature and humidity V.S. the Date in the same plot, using the following command.
`weather_data_ordered.plot(subplots=True, x= 'Formatted Date', y= ['Temperature (C)', 'Humidity'])`
13. Now try plotting different variables in the same plot for a certain year.






