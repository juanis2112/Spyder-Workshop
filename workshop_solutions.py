# -*- coding: utf-8 -*-
"""Workshop main flow."""

# pylint: disable=invalid-name, fixme


# %% [1] Importación de librerías y datos

# Importaciones de terceros
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split

# Importaciones locales
from utils import aggregate_by_year, plot_correlations, plot_color_gradients


# %% [2] Exploración de datos

# Cargar los datos
weather_data = pd.read_csv('data/weatherHistory.csv')

# Imprimir longitud de datos
len(weather_data)

# Imprimir las primeras tres filas del DataFrame
weather_data.head(3)

# TODO: Imprime las últimas tres filas de DataFrame
weather_data.tail(3)


# %% [3] Visualización

# Ordenar filas según fecha
weather_data = pd.read_csv('data/weatherHistory.csv')
weather_data['Formatted Date'] = pd.to_datetime(
    weather_data['Formatted Date'].str[:-6])
weather_data_ordered = weather_data.sort_values(by='Formatted Date')

# Restablecer índice para restaurar su orden
weather_data_ordered = weather_data_ordered.reset_index(drop=True)

# Eliminar columnas categóricas
weather_data_ordered = weather_data_ordered.drop(
    columns=['Summary', 'Precip Type', 'Loud Cover', 'Daily Summary'])

# Trazar temperatura vs. fecha
weather_data_ordered.plot(
    x='Formatted Date', y='Temperature (C)', color='red', figsize=(15, 8))

# TODO: Trace la temperatura frente a la fecha usando solo los datos de 2006
weather_data_ordered.loc[
    weather_data_ordered["Formatted Date"].dt.year == 2006, :].plot(
        x='Formatted Date', y='Temperature (C)', color='red')

# Trazar temperatura y humedad en la misma parcela
weather_data_ordered.plot(
    subplots=True, x='Formatted Date', y=['Temperature (C)', 'Humidity'],
    figsize=(15, 8))

# TODO: Grafique diferentes combinaciones de las variables y para diferentes años


# %% [4] Resumen y agregación de datos

# Datos meteorológicos por año
weather_data_by_year = aggregate_by_year(
    weather_data_ordered, date_column='Formatted Date')

# TODO: crea y usa una función para promediar los datos meteorológicos por mes

# %% [5] Análisis e interpretación de datos

# Trazar correlaciones
plot_correlations(weather_data_ordered, size=18)

# Trazar mapas de color degradados
plot_color_gradients(
    cmap_category='Plot gradients convention', cmap_list=['viridis', ])

# Calcular correlaciones
weather_correlations = weather_data_ordered.corr()
weather_data_ordered['Temperature (C)'].corr(
    weather_data_ordered['Humidity'])

# TO DO: Get the correlation for different combinations of variables.
#       Contrast them with the weather_correlations dataframe


# %% [6] Modelado y predicción de datos

# Obtener subconjuntos de datos para el modelo
x_train, x_test, y_train, y_test = train_test_split(
    weather_data_ordered['Humidity'], weather_data_ordered['Temperature (C)'],
    test_size=0.25)

# Ejecutar regresión
regression = linear_model.LinearRegression()
regression.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

# Imprimir coeficientes
print(regression.intercept_, regression.coef_)  # beta_0, beta_1


# %% [7] Prueba y evaluación de modelos predictivos

# Trazar modelo predicho con datos de prueba
y_predict = regression.predict(x_test.values.reshape(-1, 1))
plt.scatter(x_test, y_test, c='red', label='Observation', s=1)
plt.scatter(x_test, y_predict, c='blue', label='Model')
plt.xlabel('Humidity')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()

# TODO: Usando el modelo, predecir la temperatura para un nivel dado de humedad

# Evaluar el modelo numéricamente
explained_variance_score(y_test, y_predict)
