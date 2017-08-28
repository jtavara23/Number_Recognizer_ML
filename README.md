# Reconocedor De Numeros usando Redes Convolucionales
1. Introduccion

2. Procesamiento de Datos <br />
2.1 Cargar los Datos<br />
2.2 Normalizar<br />
2.3 Asignacion de clases<br />
2.4 Dividir conjunto de entrenamiento y validación<br />

3. CNN<br />
3.1 Define the model<br />
3.2 Set the optimizer and annealer<br />
3.3 Data augmentation<br />

4. Evaluate the model <br />
4.1 Training and validation curves<br />

5. Prediction and submition<br />
5.1 Predict and Submit results<br />
5.2 Matriz de confusion
## 1. Introduccion
Las redes neuronales convolucionales (CNNs) son una variación biológicamente inspirada de los perceptrones multicapa (MLPs). A diferencia de MLPs donde cada neurona tiene un vector de peso separado, las neuronas en las CNNs comparten pesos.
La implementacion de este proyecto se hizo en el lenguaje Python y usando Tensorflow, debido a que la implementacion de una CNN desde cero toma mucho tiempo, existen diversas librerias que ayudan a realizar esta tarea. (http://deeplearning.net/software_links/)<br />
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
![numeros_mnist](https://user-images.githubusercontent.com/18404919/29759317-4b838534-8b80-11e7-9533-ed582f7ef037.png)

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
classes = activation_vector(labels_flat, CLASS_COUNT)
classes = classes.astype(np.uint8)
```
### 2.4 Dividir conjunto de entrenamiento y validación
``` python
#cantidad de imagenes del conjunto de entrenamiento separadas para validar
VALIDATION_SIZE = 4000

validation_images = images[:VALIDATION_SIZE]
validation_labels = classes[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = classes[VALIDATION_SIZE:]
train_labels_flat = labels_flat[VALIDATION_SIZE:]
```
