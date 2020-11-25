# Spyder Workshop

El objetivo principal de este taller es explorar algunas de las funciones básicas de Spyder para la programación científica.
Trabajaremos en la visualización, el análisis y la predicción de datos utilizando librerías de Python como Pandas, Matplotlib y Scikit-learn sobre un conjunto de datos de observaciones meteorológicas históricas de 2006 a 2016.


## Prerequisitos

Para comenzar con este taller, necesitará tener [Spyder](https://www.spyder-ide.org) instalado en un ambiente de Python 3 que contenga al menos las librerías Numpy, Matplotlib, Pandas y Scikit-Learn.
Le recomendamos que descargue e instale la [distribución de Python Anaconda](https://www.anaconda.com/products/individual), que contiene todas estas librerías y más en un solo lugar.


## Configuración del proyecto

0. Si está familiarizado con git, clone el [repositorio de Spyder-Workshop](https://github.com/juanis2112/Spyder-Workshop):

```bash
git clone https://github.com/juanis2112/Spyder-Workshop
```

De lo contrario, puede descargar el contenido del workshop [aquí](https://github.com/juanis2112/Spyder-Workshop/archive/master.zip).
Luego, inicie Spyder a través del acceso directo del menú de inicio en Windows, o desde Anaconda Navigator en Linux o Mac.
Abra el Workshop en Spyder como un proyecto haciendo clic en `Abrir proyecto` en el menú `Proyecto` y navegando al directorio `Spyder-Workshop`.
Finalmente, abra el archivo `workshop.py` haciendo doble clic en él en el panel `Proyecto` a la izquierda de la ventana principal de Spyder.


## Importación de librerías y datos

Lo primero que debemos hacer antes de comenzar nuestro trabajo es importar las librerías necesarias para nuestro análisis y cargar los datos de una manera que sea fácil de explorarlos.

1. Importe las librerías Matplotlib y Pandas:

```python
import matplotlib.pyplot as plt
import pandas as pd
```

2. Cargue los datos del archivo CSV a un DataFrame de Pandas:

```python
weather_data = pd.read_csv('data/weatherHistory.csv')
```


## Exploración de datos

Ahora que tenemos nuestros datos y librerías listos, comencemos por echar un vistazo a los datos que tenemos.

3. Abra la variable `weather_data` en el panel del Explorador de variables haciendo doble clic en su nombre.
El Explorador de variables se encuentra en la parte superior derecha de la ventana principal de Spyder; es posible que deba hacer clic en su pestaña para que sea visible.

4. Verifique que el `Size` de `weather_data` en el Explorador de variables se corresponda con el resultado de `len(weather_data)` en la Terminal de IPython.

```python
len(weather_data)
```

5. Imprima las primeras tres filas del DataFrame en la Terminal de IPython:

```python
weather_data.head(3)
```

6. Ahora, intente imprimir las últimas tres filas del DataFrame.


## Visualización

Una herramienta útil para explorar los datos con los que vamos a trabajar es graficarlos.
Esto es fácil de hacer usando la librería de pandas, que importamos previamente.

Lo primero que queremos hacer antes de graficar nuestros datos es ordenar las filas según la fecha.
Utilice el Explorador de variables para verificar que nuestros datos no están ordenados de forma predeterminada.

7. Formateé la fecha y cree un nuevo DataFrame con nuestros datos ordenados por el mismo

```python
weather_data['Formatted Date'] = pd.to_datetime(
    weather_data['Formatted Date'].str[:-6])
weather_data_ordered = weather_data.sort_values(by='Formatted Date')
```

8. En el Explorador de variables, haga clic con el botón derecho en el antiguo DataFrame `weather_data` para que aparezca el menú contextual y seleccione `Eliminar` para eliminarlo.
Ahora, vamos a trabajar con nuestra nueva variable `weather_data_ordered`.

Observe en el Explorador de variables que el índice del DataFrame (la columna `Índice` a la izquierda) no está en el orden de la fecha.
Restablezca el índice para que su orden coincida con el de `Formatted Date`:

```python
weather_data_ordered = weather_data_ordered.reset_index(drop=True)
```

También vemos que existen algunas variables cualitativas que pueden dificultar nuestro análisis.
Por esta razón, queremos ceñirnos a las columnas que nos dan información numérica y descartar las categóricas:

```python
weather_data_ordered = weather_data_ordered.drop(
    columns=['Summary', 'Precip Type', 'Loud Cover', 'Daily Summary'])
```

9. Grafique `Temperature (C)` versus `Formatted Date` para ver cómo cambia la temperatura con el tiempo:

```python
weather_data_ordered.plot(
    x='Formatted Date', y='Temperature (C)', color='red', figsize=(15, 8))
```

10. Cambie al panel de Gráficos (Plots), en la misma sección superior derecha de la interfaz que el Explorador de variables, para ver su figura.

11. Ahora, intente graficar la temperatura versus la fecha utilizando solo los datos de 2006.

12. Grafique la temperatura y la humedad versus la fecha en el mismo gráfico para examinar cómo cambian ambas variables con el tiempo:

```python
weather_data_ordered.plot(
    subplots=True, x='Formatted Date', y=['Temperature (C)', 'Humidity'],
    figsize=(15, 8))
```

13. Ahora, intente graficar diferentes variables en la misma gráfica para diferentes años.


## Resumen y agregación de datos

Los gráficos anteriores contenían una gran cantidad de datos, lo que dificulta la comprensión de la evolución de nuestras variables a lo largo del tiempo.
Por esta razón, una de las cosas que podemos hacer es agrupar la información que tenemos por año y graficar los valores anuales.
Para hacer esto, hemos escrito una función en el archivo `utils.py`, en la misma carpeta que este taller, que crea una nueva columna en el DataFrame que contiene el año, y luego agrupa los valores por año, calculando el promedio del variables para cada uno.

14. Importe la función del módulo `utils`, para que pueda usarla en su archivo:

```python
from utils import aggregate_by_year
```

15. Úsela para agregar por año y graficar los datos:

```python
weather_data_by_year = aggregate_by_year(
    weather_data_ordered, date_column='Formatted Date')
```

16. Intente escribir una función en el archivo `utils.py` que obtenga los promedios de los datos meteorológicos por mes y los grafique.


## Análisis e interpretación de datos

Ahora, queremos evaluar las relaciones entre las variables en nuestro conjunto de datos.
Para ello, hemos escrito otra función en `utils.py`.

17. Importar la nueva función:

```python
from utils import aggregate_by_year, plot_correlations
```

18. Grafique las correlaciones entre las variables y vea la figura en el panel de gráficos:

```python
plot_correlations(weather_data_ordered, size=18)
```


19. Importe la función `plot_color_gradients()`, que le ayudará a trazar el gradiente del mapa de colores para poder interpretar su diagrama de correlación:

```python
from utils import aggregate_by_year, plot_correlations, plot_color_gradients
```

20. Grafique el gradiente del mapa de colores con la función que importó:

```python
plot_color_gradients(
    cmap_category='Plot gradients convention', cmap_list=['viridis', ])
```

21. Calcule las correlaciones de Pearson entre las diferentes variables de nuestro conjunto de datos:

```python
weather_correlations = weather_data_ordered.corr()
```

22. Abra la variable `weather_correlations` en el Explorador de variables para ver los resultados.

23. Imprima la correlación entre la humedad y la temperatura en la Terminal de IPython:

```python
weather_data_ordered['Temperature (C)'].corr(weather_data_ordered['Humidity'])
```

Verifique que tenga el mismo valor que en el DataFrame `weather_correlations`.

24. Intente calcular correlaciones entre diferentes variables y compararlas con las del DataFrame.


## Modelado y predicción de datos

Finalmente, queremos usar nuestros datos para construir un modelo que nos permita predecir valores para algunas de nuestras variables.
En la sección anterior, nos dimos cuenta de que la humedad y la temperatura son dos de las variables más correlacionadas, por lo que vamos a utilizar estas dos primero.

Vamos a utilizar Scikit-Learn, que es una librería de Python que contiene herramientas para explorar datos y construir diferentes tipos de modelos predictivos.

25. Importe los dos objetos necesarios para nuestro modelado de datos:

```python
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```

26. Una forma clásica de hacer un modelo predictivo es subdividir los datos en dos conjuntos: entrenamiento y prueba.
Los datos de entrenamiento nos ayudarán a ajustar nuestro modelo predictivo, mientras que los datos de prueba jugarán el papel de observaciones futuras y nos darán una idea de cuán buenas son nuestras predicciones.

Scikit-Learn contiene una función incorporada para dividir sus datos:

```python
x_train, x_test, y_train, y_test = train_test_split(
    weather_data_ordered['Humidity'], weather_data_ordered['Temperature (C)'],
    test_size=0.25)
```

27. Usaremos la regresión lineal en Scikit-Learn para hacer un modelo lineal de nuestros datos.
Cree el modelo, luego ajústelo con los datos meteorológicos:

```python
regression = linear_model.LinearRegression()
regression.fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
```

28. Coloque el cursor de texto sobre `LinearRegression()` y presione el acceso directo `Inspect` (`Ctrl+I` por defecto en Windows/Linux, o `Cmd-I` en macOS) para obtener la documentación de esta función en el Panel de ayuda .

29. Imprima los coeficientes de nuestra regresión:

```python
print(regression.intercept_, regression.coef_)  # beta_0, beta_1
```

Tenga en cuenta que esto significa que nuestro modelo es una función lineal `$$y = beta_0 + beta_1 \times x$$`, donde la temperatura es una función de la humedad.


## Prueba y evaluación de modelos predictivos

30. Ahora, queremos graficar las predicciones de nuestro modelo frente a nuestros datos de prueba, para ver qué tan buenas fueron nuestras predicciones:

```python
y_predict = regression.predict(x_test.values.reshape(-1, 1))
plt.scatter(x_test, y_test, c='red', label='Observation', s=1)
plt.scatter(x_test, y_predict, c='blue', label='model')
plt.xlabel('Humidity')
plt.ylabel('Temperature (C)')
plt.legend()
plt.show()
```

31. Usando el método `.predict()` de nuestro modelo, prediga la temperatura para un nivel dado de humedad.

32. Por último, podemos evaluar numéricamente qué tan buenas fueron las predicciones de nuestro modelo.
Para esto, usaremos la función `explained_variance_score` disponible en `sklearn.metrics`.
Esta métrica se calcula como `$$1-(Var(Y_real-Y_model)/Var(Y_real))$$`, lo que significa que cuanto más cerca esté el valor de 1, mejor será nuestro modelo.

Necesitamos importar la función que evalúa nuestro modelo:

```python
from sklearn.metrics import explained_variance_score
```

33. Calcule la puntuación de varianza explicada e imprímala:

```python
explained_variance_score(y_test, y_predict)
```
