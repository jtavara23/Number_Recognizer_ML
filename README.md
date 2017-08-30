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

### 3.2 Construccion de La red Convolucional
#### 3.2.1 Estructura del modelo

<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29766434-582acb70-8ba4-11e7-8efb-2c9e192d202d.png" />
</p>

**Configuracion de Parametros**<br/>
Los tensores placeholder sirven como entrada al gráfico computacional de TensorFlow que podemos cambiar cada vez que ejecutamos el gráfico.
'None' significa que el tensor puede contener un número arbitrario de imágenes, donde cada imagen es un vector de longitud dada.
``` python
# imagenes
x = tf.placeholder('float', shape=[None, tam_imagen],name= NOMBRE_TENSOR_ENTRADA)
# clases
y_ = tf.placeholder('float', shape=[None, CANT_CLASES],name= NOMBRE_TENSOR_SALIDA_DESEADA)
```
Las entradas y salidas para las capas convolucionales se hacen a traves de tensores de 4 dimensiones:
  1. Cantidad images
  2. Altura de imagen
  3. Anchura de imagen
  4. Canales por imagen(de color/ de filtro)
``` python
#las capas de convolucion esperan que las entradas sean encodificadas en tensores de 4D
imagen = tf.reshape(x, [-1,altura_imagen, anchura_imagen,1])
print (imagen.get_shape()) # =>(56000,28,28,1)
```

**Fuciones de Inicializacion**<br/>
``` python
# tf.truncated_normal: Outputs random values from a truncated normal distribution.
# The generated values follow a normal distribution with specified standard deviation.
def inicializar_pesos(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# tf.constant: Creates a constant tensor. 
# The resulting tensor is populated with values of type dtype, as specified by arguments value and shape
def inicializar_bias(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
``` 
**Primera Capa Convolucional**
``` python
#[tamanho_filtro,tamanho_filtro, canalEnt_img, cant_filtros]
forma = [5, 5, 1, 32]
pesos_conv1 = inicializar_pesos(shape = forma)
biases_conv1 = inicializar_bias([32])

#strides=[img,movx,movy ,filtro]
convolucion1 = tf.nn.conv2d(imagen,
                         pesos_conv1,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
print (convolucion1.get_shape()) # => (56000, 28,28, 32)

#el valor del bias es adicionado para cada resultado de convolucion
convolucion1 += biases_conv1

#funcion de activacion
act_conv1 = tf.nn.relu(convolucion1)
print(act_conv1.get_shape()) # => (56000, 28, 28, 32)

pool_conv1 = tf.nn.max_pool(act_conv1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
print(pool_conv1.get_shape()) # => (56000, 14, 14, 32)
``` 
**Segunda Capa Convolucional**
``` python
#Forma de los pesos del filtro
#[tamanho_filtro,tamanho_filtro, canalEnt_img, cant_filtros]
forma = [5, 5, 32, 64]
pesos_conv2 = inicializar_pesos(shape = forma)
biases_conv2 = inicializar_bias([64])

#strides=[img,movx,movy ,filtro]
convolucion2 = tf.nn.conv2d(pool_conv1,
                         pesos_conv2,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
print (convolucion2.get_shape()) # => (56000, 14,14, 64)

#el valor del bias es adicionado para cada resultado de convolucion
convolucion2 += biases_conv2

#funcion de activacion
act_conv2 = tf.nn.relu(convolucion2)
print (act_conv2.get_shape()) # => (56000, 14,14, 64)

pool_conv2 = tf.nn.max_pool(act_conv2,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
print (pool_conv2.get_shape()) # => (56000, 7, 7, 64)
``` 
**Capa Totalmente Conectada(Fully Connected)**<br/>
Las entradas y salidas para las capas FC se hacen a traves de tensores de 2 dimensiones:
  1. Cantidad Datos entrada
  2. Cantidad Datos salida
``` python
forma = [7 * 7 * 64, 1024]
pesos_fc1 = inicializar_pesos(shape = forma)
biases_fc1 = inicializar_bias([1024])

pool_conv2_flat = tf.reshape(pool_conv2, [-1, 7*7*64])
# (56000, 7, 7, 64) => (56000, 3136)

# Multiplicar matriz 'pool_conv2_flat' por matriz 'pesos_fc1' y sumar bias
fc1 = tf.matmul(pool_conv2_flat, pesos_fc1) + biases_fc1
#print (fc1.get_shape()) # => (56000, 1024)

act_fc1 = tf.nn.relu(fc1)
``` 

**Aplicar Dropout**
>Dropout es una técnica de regularización con el objetivo de reducir el sobreajuste que puede darse durante el entrenamiento de una rede neuronal. Consiste en asignar algunas neuronas a cero,para evitar que la red al entrenarse dependa demasiado en estas neuronas. Con esto se producen modelos neuronales robustos.
<p align="center">
<img src = "https://user-images.githubusercontent.com/18404919/29856355-e26d11d4-8d17-11e7-8abe-181b61fdd362.png"  width="480" />
</p>

``` python
#Creamos un placeholder para que la probabilidad de la salida de una neurona se mantenga durante el dropout. Esto nos permite activar el dropout durante el entrenamiento y desactivarlo durante las pruebas.
keep_prob = tf.placeholder('float',name=NOMBRE_PROBABILIDAD)
#Aplicarmos dropout entre la capa FC y la capa de salida
h_fc1_drop = tf.nn.dropout(act_fc1, keep_prob)
``` 

**Capa de Salida**<br/>
Estima la probabilidad de que la imagen de entrada pertenezca a cada una de las 10 clases. Sin embargo, estas estimaciones son un poco difíciles de interpretar porque los números pueden ser muy pequeños o grandes, por lo que queremos normalizarlos para que cada elemento esté limitado entre cero y uno y los 10 elementos sumen a uno. Esto se calcula utilizando la función softmax y el resultado se almacena en y_calculada.
``` python
pesos_fc2 = inicializar_pesos([1024, CANT_CLASES])
biases_fc2 = inicializar_bias([CANT_CLASES])
fc2 = tf.matmul(h_fc1_drop, pesos_fc2) + biases_fc2

y_calculada = tf.nn.softmax(fc2, name = NOMBRE_TENSOR_SALIDA_CALCULADA)
print (y_calculada.get_shape()) # => (56000, 10)

#El número de clase es el índice del elemento más grande.
#[0.01, 0.04, 0.02, 0.5, 0.03 0.01, 0.05, 0.02, 0.3, 0.02] => 3
clase_predecida = tf.argmax(y_calculada,dimension = 1)
tf.add_to_collection("predictor", clase_predecida)	
``` 

**Funcion de Costo de Error**<br/>
Definimos la función de costo de error para medir cuán mal desempeña nuestro modelo en imágenes con sus clases conocidas. El costo que queremos minimizar va a estar en función de lo calculado con lo deseado(real).
``` python
costo = -tf.reduce_sum(y_deseada * tf.log(y_calculada))
``` 
Y para minimizar este costo de error usamos el optimizador ADAM(es adecuado para problemas con muchos parámetros). Esta función mejorará iterativamente los parámetros(valores de los filtros[pesos] y bias de las neuronas)
``` python
#Creamos un placeholder para guardar las optimizaciones durante las iteraciones del entrenamiento
iterac_entren = tf.Variable(0, name='iterac_entren', trainable=False)
``` 
Para la optimizacion se requiere que se inicialice con una **tasa de aprendizaje**.Esta determina la rapidez o la lentitud con que desea actualizar los parámetros. Por lo general, uno puede comenzar con una gran tasa de aprendizaje, y disminuir  la tasa de aprendizaje a medida que progresa la formación.
``` python
#TASA_APRENDIZAJE = 5e-4  #hasta 3000 iteraciones
#TASA_APRENDIZAJE = 1e-4  #desde 3000 a (+)
optimizador = tf.train.AdamOptimizer(TASA_APRENDIZAJE).minimize(costo, global_step=iterac_entren)
``` 

**Evaluacion de acierto**<br/>
``` python
#Con un vector de booleanos sabremos si la clase calculada es igual a la clase verdadera de cada imagen.
prediccion_correcta = tf.equal(tf.argmax(y_calculada,1), tf.argmax(y_deseada,1))
#Se calcula la precisión de clasificación convirtiendo los datos boolenados a flotantes
#de modo que False se convierte en 0 y True se convierte en 1, para luego calcular el promedio de estos números.
acierto = tf.reduce_mean(tf.cast(prediccion_correcta, 'float'))
``` 

