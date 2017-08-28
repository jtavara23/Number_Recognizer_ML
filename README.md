# Reconocedor De Numeros usando Redes Neuronales Convolucionales
1. Introduccion

2. Procesamiento de Datos <br />
2.1 Cargar los Datos<br />
2.2 Normalizar<br />
2.3 Asignacion de clases<br />
2.4 Dividir conjunto de entrenamiento y validación<br />

3. Red Convolucional<br />
3.1 Estructura del modelo<br />
3.2 Set the optimizer and annealer<br />
3.3 Data augmentation<br />

4. Evaluate the model <br />
4.1 Training and validation curves<br />

5. Prediction and submition<br />
5.1 Predict and Submit results<br />
5.2 Matriz de confusion

## 1. Introduccion
Si se desea aplicar el redes neuronales para el reconocimiento de imágenes, las redes neuronales convolucionales (CNN) es el camino a seguir. Ha estado barriendo el tablero en competiciones por los últimos años, pero quizás su primer gran éxito vino en los últimos 90's cuando Yann LeCun lo utilizó para resolver MNIST con el 99.5% de exactitud.<br/>
Usando una red simple totalmente conectada (sin convolución) se podria alcanzar el 95-96%, lo cual no es muy buen resultado en este conjunto de datos. En contraste, la implementacion hecha en este proyecto es casi el estado del arte,llegando a obtener un **99.3%** de acierto <br/>
La implementacion de este proyecto se realizó en el lenguaje Python.<br /> 
Para la implementacion de la CNN se utilizó Tensorflow, debido a que la implementacion de una CNN desde cero toma mucho tiempo, existen diversas librerias que ayudan a realizar esta tarea. (http://deeplearning.net/software_links/)<br />
Para el proceso de procesamiento de imagenes se utilizo la libreria OpenCV.

### Librerias Usadas:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from datetime import timedelta
from funcionesAuxiliares import  display,activation_vector
```
## 2. Procesamiento de Datos
Link: http://yann.lecun.com/exdb/mnist/
>La base de datos MNIST("Modified National Institute of Standards and Technology") de dígitos manuscritos, disponible en esta página, tiene un conjunto de entrenamiento de 60.000 ejemplos y un conjunto de prueba de 10.000 ejemplos. Es un subconjunto de un conjunto más grande disponible de NIST. Los dígitos se han normalizado de tamaño y se han centrado en una imagen de tamaño fijo.
Es una buena base de datos para las personas que quieren probar técnicas de aprendizaje y métodos de reconocimiento de patrones en los datos del mundo real, ya que se evita el esfuerzo de preprocesar y formatear las imagenes.<br />
<p align="center">
  <img src=https://user-images.githubusercontent.com/18404919/29759317-4b838534-8b80-11e7-9533-ed582f7ef037.png>
</p>

###  2.1 Cargar los Datos 
```python 
datasetTraining = pd.read_csv(path+'datasets/60ktrain.csv')
images = datasetTraining.iloc[:,1:].values
images = images.astype(np.float)
```

###  2.2 Normalizar
```python 
# Normalizar, convertir de [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)	
```
### 2.3 Asignacion de clases
Para la mayoría de los problemas de clasificación, se utilizan "vectores de activacion". Un vector de activacion es un vector que contiene un único elemento igual a 1 y el resto de los elementos igual a 0. En este caso, el n-ésimo dígito se representa como un vector cero con 1 en la posición n-ésima.<br />
```python 
#Organizar las clases de las imagenes en un solo vector
labels_flat = datasetTraining.iloc[:,0].values.ravel()
# convertir tipo de clases de escalares a vectores de activacion de 1s
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
classes = activation_vector(labels_flat, CLASS_COUNT)
classes = classes.astype(np.uint8)
```
```python
def activation_vector(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
```
### 2.4 Dividir conjunto de entrenamiento y validación
Por último, reservamos algunos datos para su validación. Es esencial en modelos de ML tener un conjunto de datos independiente que no participa en el entrenamiento y se utiliza para asegurarse de que lo que hemos aprendido en realidad se puede generalizar.
``` python
#cantidad de imagenes del conjunto de entrenamiento separadas para validar
VALIDATION_SIZE = 4000

validation_images = images[:VALIDATION_SIZE]
validation_labels = classes[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = classes[VALIDATION_SIZE:]
train_labels_flat = labels_flat[VALIDATION_SIZE:]
```
## 3. Red Convolucional
Las redes neuronales convolucionales (CNNs) son una variación biológicamente inspirada de los perceptrones multicapa (MLPs). A diferencia de MLPs donde cada neurona tiene un vector de peso separado, las neuronas en las CNNs comparten pesos.<br />
Utilizando la estrategia de compartir de pesos, las neuronas son capaces de realizar convoluciones en los pixels de una imagen utilizando un **filtro de convolución(kernel)** el cual esta formado por pesos.</br> 
>Fitro de Convolucion
<img src="https://user-images.githubusercontent.com/18404919/29761167-91551f52-8b8d-11e7-815b-aaac24408588.png" heigth="480" width="480">
Las redes convolucionales funcionan moviendo pequeños filtros a través de la imagen de entrada. Esto significa que los filtros se reutilizan para reconocer patrones en toda la imagen de entrada. Esto hace que las Redes Convolucionales sean mucho más potentes que las Redes Completamente Conectadas con el mismo número de variables. Esto a su vez hace que las Redes Convolucionales sean más rápidas para entrenar.



Esto es seguido por una operación de agrupación que como forma de muestreo descendente no lineal, reduce progresivamente el tamaño espacial de la representación reduciendo así la cantidad de cálculo y los parámetros en la red.

