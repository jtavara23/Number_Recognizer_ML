# Reconocedor De Numeros usando Redes Convolucionales
1. Introduccion

2. Procesamiento de Datos <br />
2.1 Cargar los Datos<br />
2.2 Normalizar<br />
2.3 Redimensionar<br />
2.4 Asignacion de clases<br />
2.5 Dividir conjunto de entrenamiento y validación<br />

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

## 2. Procesamiento de Datos
Link: http://yann.lecun.com/exdb/mnist/
>La base de datos MNIST("Modified National Institute of Standards and Technology") de dígitos manuscritos, disponible en esta página, tiene un conjunto de entrenamiento de 60.000 ejemplos y un conjunto de prueba de 10.000 ejemplos. Es un subconjunto de un conjunto más grande disponible de NIST. Los dígitos se han normalizado de tamaño y se han centrado en una imagen de tamaño fijo.
Es una buena base de datos para las personas que quieren probar técnicas de aprendizaje y métodos de reconocimiento de patrones en los datos del mundo real, ya que se evita el esfuerzo de preprocesar y formatear las imagenes.
![numeros_mnist](https://user-images.githubusercontent.com/18404919/29759317-4b838534-8b80-11e7-9533-ed582f7ef037.png)
###  2.1 Cargar los Datos 
```datasetTraining = pd.read_csv(path+'datasets/60ktrain.csv')```
