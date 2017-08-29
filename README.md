# Reconocedor De Numeros usando Redes Neuronales Convolucionales
1. Introduccion

2. Procesamiento de Datos <br />
2.1 Cargar los Datos<br />
2.2 Normalizar<br />
2.3 Asignacion de clases<br />
2.4 Dividir conjunto de entrenamiento y validación<br />

3. Red Convolucional<br />
3.1 Conceptos basicos
3.2 Estructura del modelo<br />

4. Evaluate the model <br />
4.1 Training and validation curves<br />

5. Prediction and submition<br />
5.1 Predict and Submit results<br />
5.2 Matriz de confusion

## 1. Introduccion
Si se desea aplicar el redes neuronales para el reconocimiento de imágenes, las redes neuronales convolucionales (CNN) es el camino a seguir. Ha estado barriendo el tablero en competiciones por los últimos años, pero quizás su primer gran éxito vino en los últimos 90's cuando Yann LeCun lo utilizó para resolver MNIST con el 99.5% de exactitud.<br/>
Usando una red simple totalmente conectada (sin convolución) se podria alcanzar el 90-95%, lo cual no es muy buen resultado en este conjunto de datos. En contraste, la implementacion hecha en este proyecto es casi el estado del arte,llegando a obtener un **99.3%** de acierto <br/>
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
### 3.1 Conceptos Basicos
Las redes neuronales convolucionales (CNNs) son una variación biológicamente inspirada de los perceptrones multicapa (MLPs). A diferencia de MLPs donde cada neurona tiene un vector de peso separado, las neuronas en las CNNs comparten pesos.<br />
Utilizando la estrategia de compartir de pesos, las neuronas son capaces de realizar **convoluciones** en los pixels de una imagen utilizando un **filtro de convolución(kernel)** el cual está formado por pesoss.</br> 

**Fitro de Convolucion(KERNEL)**

<p align="center">
<img src="https://user-images.githubusercontent.com/18404919/29761167-91551f52-8b8d-11e7-815b-aaac24408588.png" width="480">
</p>

**Convolucion** 
>Es el proceso que consiste en calcular la coincidencia de un kernel con una parte de la imagen,y para conseguirlo simplemente se multiplica cada píxel en el kernel por el valor del píxel en la imagen. Para luego, sumar las respuestas y dividirlas por el número total de píxeles en el kernel.<br/>
Para completar la convolución en toda la imagen, repetimos este proceso, alineando el kernel con cada parte de imagen posible. El resultado es una versión filtrada de nuestra imagen original.
<p float="left">
<img src = "https://user-images.githubusercontent.com/18404919/29762130-6b002a04-8b92-11e7-8933-5198ac33665d.png"  width="400" hspace="20" />
<img src = "https://user-images.githubusercontent.com/18404919/29762538-8abcc3be-8b94-11e7-9bc2-11ce2f359ac4.png" width="400" height = 200/>
</p>

>El siguiente paso es repetir el proceso de convolución no solo para un tipo de filtro(kernel) sino para varios. El resultado es un conjunto de imágenes filtradas, una para cada uno de nuestros filtros. Es conveniente pensar en toda esta colección de operaciones de convolución como un único paso de procesamiento. En CNNs esto se conoce como una capa de convolución, haciendo alusión al hecho de que pronto tendrá otras capas agregadas a ella.
<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29763319-3c606b40-8b98-11e7-8f6e-d73977680d20.png" width="480" >
</p>

Las redes convolucionales funcionan moviendo estos pequeños filtros(kernels) a través de la imagen de entrada. Esto significa que los filtros se reutilizan para reconocer patrones en toda la imagen de entrada. Esto hace que las Redes Convolucionales sean mucho más potentes que las Redes Completamente Conectadas con el mismo número de variables.

<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29763436-b07532d6-8b98-11e7-87de-e3d91c853947.png" width="480" >
</p>

**Pooling**
>Es otra tecnica poderosa que utilizan las CNNs. Pooling(agrupacion) es una manera de tomar imágenes grandes y reducirlas mientras conserva la información más importante en ellas(esto reduce así la cantidad de cálculo y los parámetros en la red). El proceso matematico consiste en pasar una pequeña ventana através de una imagen y tomar el valor máximo de la ventana en cada paso.
<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29763741-0e66f13a-8b9a-11e7-8037-117d23a87ab2.png"  width="480" />
</p>

>Debido a que mantiene el valor máximo de cada ventana, conserva los mejores ajustes de cada característica dentro de la ventana. Esto significa que no le importa tanto exactamente donde se ajuste la característica, siempre y cuando se ajuste en algún lugar dentro de la ventana. El resultado de esto es que CNNs puede encontrar si una característica está en una imagen sin preocuparse exactamente de donde está. Esto ayuda a resolver el problema de las computadoras al comparar imagenes de manera hiper-literal.
<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29763118-6107ce4e-8b97-11e7-98e8-14a124f1d7e8.png"  width="480" />
</p>

**RELU(Rectified Linear Units)**
>Entre la capa de convolucion y la capa de pooling se encuentra la capa RELU que esta compuesta por neuronas que poseen una funcion de activación llamada **Función lineal rectificada** que deriva de la función de activación sigmoidal, pero tiene mayores ventajas que esta última y tambien de la tangencial.(http://www.jefkine.com/general/2016/08/24/formulating-the-relu/) 
<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29765002-05919060-8b9f-11e7-94c5-3943d3b0ca4f.png"  width="480" />
</p>

>Cuando se produce un número negativo, se intercambia por un 0. Esto ayuda a que la CNN se mantenga matemáticamente sana al manteniendo a los valores aprendidos de quedar cerca de 0 o mayor a este, y así asegura que la salida sea siempre positiva porque los valores negativos se ponen a cero.

<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29765017-11f9a96e-8b9f-11e7-9f61-f26b5184f499.png"  width="480" />
</p>

Las imágenes se filtran(capa_Convolucion), se rectifican(capa_RELU) y se agrupan(capa_Pooling) para crear un conjunto de imágenes reducidas y filtradas por características. Estos pueden ser filtrados y encogidos una y otra vez. Cada vez, las características se hacen más grandes y más complejas, y las imágenes se vuelven más compactas.

**MLP Totalmente Conectado**
>Eventualmente, con un mapa de características lo suficientemente pequeño, el contenido se aplastará en un vector de una dimensión y será entrada para en un MLP totalmente conectado para su procesamiento. La última capa de este MLP totalmente conectado es visto como la salida.<br/>
Cuando se presenta una nueva imagen a la CNN, se filtra a través de las capas inferiores hasta que alcanza al final la capa totalmente conectada. Luego se lleva a cabo una elección y la respuesta con la mayoría de los votos gana y se declara la categoría de la entrada.
<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29766302-eb5cc8d6-8ba3-11e7-9426-e2e9ae9bd8bf.png"  width="480" />
</p>
Para mas información, visitar [aquí] (http://brohrer.github.io/how_convolutional_neural_networks_work.html)

### 3.2 Estructura del modelo
<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29766434-582acb70-8ba4-11e7-8efb-2c9e192d202d.png" />
</p>
