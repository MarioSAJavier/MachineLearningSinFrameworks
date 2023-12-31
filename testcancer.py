# -*- coding: utf-8 -*-
"""TestCancer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kB3_shjMs9QZ_vhoXasDgYSQkJZu3j61

# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación) [semana 3]

**Mario Javier Soriano Aguilera A01384282**
"""

from google.colab import drive
drive.mount("/content/gdrive")
!pwd

!cp /content/gdrive/MyDrive/InteligenciaArtificial/MachineLearning/LogReg/MachineLearning.py .

import pandas as pd
import numpy as np
import MachineLearning
from MachineLearning import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set()

"""El data set a utilizar contiene informacion sobre caracteristicas del cancer y el tipo de diagnostico en el que se registro."""

df = pd.read_csv('/content/gdrive/MyDrive/InteligenciaArtificial/MachineLearning/LogReg/Cancer_Data.csv')

df.head()

"""Limpieza de datos. Se observa que mis datos tienen 33 columnas y 569 filas. Ademas de contener valores nulos en toda una columna que ni siquiera tiene nombre, la cual se eliminara por que no brinda valor alguno"""

len(df)

df.isna().sum()

df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

"""Se observa el tipo de dato y vemos que tenemos la variable diagnosis que es tipo object y todos los demas valores numericos float64, usaremos la variable categorica diagnosis como nuestro target de prediccion."""

df.dtypes

df.diagnosis.unique()

df.diagnosis.value_counts().plot(kind='bar', figsize=(3,4))
plt.title("Number of cars by make")
plt.ylabel('Number of cars')
plt.xlabel('Make of the cars')

df.groupby('diagnosis').size().plot(kind='pie', autopct="%0.1f %%",labels = ['B', 'M'])

df.describe()

Tcorrelation = df.corr(method='pearson')
Tcorrelation

c = df.corr()
threshold = .80
np.abs(c.values) > threshold
[f"{c.columns[i]} and {c.columns[j]}" for i, j in zip(*np.where(np.abs(c.values) > threshold)) if i < j]

"""En lo anterior basicamente solo se realizo una matriz de correlacion ya que tenemos muchos labels y solo queremos valores significativos, de forma rapida se analizaron con una matris de correlacion, se realizo una matriz booleana de esa matriz de correlacion y se le pusieron las etiquetas para poder identificar a simple vista cuales son las variables y con cual otra tienen alta correlacion, en este caso se escogio correlacion de .7, y se escogieron las que mas se repetian ya que son las que mas variables pueden representar."""

labels = df[['radius_mean', 'perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean', 'texture_mean']]

X = labels
y = df['diagnosis'] = np.where(df['diagnosis']=='M',1,0)



#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = LogisticRegression(learningRate=.01, nIterations=1000)

clf.fit(X_train, y_train)

loss_history = clf.get_loss_history()
for epoch, loss in enumerate(loss_history):
    print(f'Época {epoch + 1}: Pérdida = {loss}')

y_pred = clf.predict(X_test)

def accuracy(yPredicted, yReal):
   return np.sum(yPredicted == yReal)/len(yReal)

acc = accuracy(y_pred, y_test)
print(acc)

"""Se obtuvo un accuracy del .83 de forma obvia hay cuestiones a mejorar ya que se realizo un analisis superficial a los datos y seria buena idea realizar un analisis estadistico mas profundo sobre las variables a emplear en nuestro modelo para que pueda disernir mejor nuestros datos. Como por ejemplo ver como se distribuyen, boxplot, variables con colinealidad a nuestro modelo, pruebas de normalidad, tranformaciones, etc.

# Explicación del Modelo

El código (from MachineLearning import LogisticRegression )define una clase de regresión logística que tiene los siguientes métodos:

**sigmoid**: es una función que calcula el valor de la función sigmoide para un número x. La función sigmoide es una función que toma valores entre 0 y 1 y se usa para modelar la probabilidad de una clase binaria, en este caso M (maligno) y B (benigno) que son como se clasifican a los tipos de cancer dependiendo de sus caracteristicas y riesgos que pueda causar al que lo porta.

**init**: es el constructor de la clase que inicializa los atributos de la instancia, como la tasa de aprendizaje, el número de iteraciones, los pesos y el sesgo del modelo. Los pesos y el sesgo son los parámetros que se van a aprender mediante el ajuste del modelo a los datos, osea el fit.

**fit**: es el método que ajusta el modelo a los datos de entrenamiento X y y. Para ello, utiliza un algoritmo de descenso de gradiente que actualiza los pesos y el sesgo en cada iteración, minimizando la función de costo logística. El método también calcula la pérdida promedio en cada iteración. En este caso se estableció que los pesos empiecen en ceros.

**predict**: es el método que hace predicciones para nuevos datos X. Para ello, calcula la regresión lineal con los pesos y el sesgo aprendidos, y luego aplica la función sigmoide para obtener las probabilidades de cada clase. Luego, asigna una etiqueta de clase 0 o 1 según si la probabilidad es menor o mayor que 0.5, respectivamente. El método devuelve una lista de predicciones de clase.

**accuracy**: calcula la precisión de las predicciones de un modelo de clasificación, mide la cantidad de valores predecidos que fueron acertados con los reales, y se divide entre el numero de valores totales.

# ¿Por qué Regresion logistica?
En lo personal me interesaba aprender la formula y llevar acabo un modelo no tan simple pero tampoco tan complicado para poder llevarlo a practica, osea se escogio primero el modelo y luego se busco un dataset adecuado para poder hacer la implementacion.

# ¿Por qué se dice que el dataset es adecuado para Regresion Logistica?
Porque este tipo de modelo es utilizado cuando se quiere hacer una clasificacion binaria, osea mi variable respuesta tiene que ser categorica y binaria, en este caso cancer maligno o benigno, ademas de tener  labels numericos con los que el modelo de regresion logistica puede utilizar para relaizar mi clasificacion. Tambien puede hacer predicciones teniendo como variables explicativas categoricas, y en ese caso se tendrian que emplear tecnicas como variables dummies, las cuales indican la presencia o ausencia de cada categoria.
"""