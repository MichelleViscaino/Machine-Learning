# Conductor Imprudente o no?
Este repositario contiene todos los archivos utilizados para crear un sistema que analiza el comportamiento de un conductor basado en la detección de señales de tránsito utilizando Deep Learning y el análisis de velocidad. Lo que sigue es un pequeño tutorial para entender que arhivos son importantes y como utilizarlos. La columna vertebral del proyecto se basa en el API de detección de objectos de Tensor Flow para Windows. Este nos ayuda a entrenar un clasificador de multiples objetos utilizando herramientas de deep learning para posteriormente poder utilizarlo para aplicaciones en tiempo real. Es por esto que el primer paso es aprender a utilizar este API y después se procederá a explicar la aplicación que se le dió para la detección de señales de tránsito.

NOTA: A manera de ayuda, el siguiente enlace direcciona a un tutorial con un proyecto similar que explica muy detalladamente la manera de usar el API de detección, con énfasis en el entrenamiento.

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

# Como utilizar el Código

* Para que funcione el sistema se necesitan todas las carpetas y archivos que se generan por la descarga del API de detección. La carpeta "utils" contiene dos archivos importantes para nuestra aplicación. Estos archivos son "label_map_util" y "visualization_utils". El segundo archivo fue modificado para nuestra aplicación por lo que si se descarga la versión original del API no funcionará el sistema. La modificación se debe a que para la aplicación se requiere retornar variables de la detección que por default no son entregados por el API ya que éste solo visualiza la detección, y el sistema requiere manejar los datos de ubicación y clase.

* La carpeta "inference_graph" es la carpeta que más espacio abarcará ya que contiene el modelo del clasificador de objetos entregado por el API de tensorflow.

* El archivo "filename2.pkl" contiene el modelo del clasificador de velocidad.

* Se necesita crear la carpeta "test_images" que contiene las imagenes con las que se probará el sistema. Dentro de ella existen dos carpetas, una para guardar las imagenes que servirán como GroundTruth para determinar la matriz de confusión de la detección y la otra para probar todo el sistema durante una trama de conducción. La carpeta con el GroundTruth debe contener un archivo .csv con las etiquetas y la carpeta de prueba debe contener un archivo .xls con la trama de velocidad almacenada durante la conducción. 
* El script "Confusion_Matrix.py" sirve para determinar la matriz de confusión de la etapa de detección de señales.
* El script "Main.py" sirve para ejecutar el análisis de conducción. Este a su vez llama a varios otros scripts dentro de la carpeta "utils" y de este repositorio.

* Para poder ejecutar el código, es necesario tener instalado tensorflow y antes de ejecutarlo se debe asegurar de activar un ambiente de tensorflow y que asi se active la GPU con el comando: `C:\> activate tensorflow1`
Si se quiere ejecutar el código se debe considerar que el directorio por defecto con el que están configurados los programas es el siguiente: C:/> tensorflow1\models\research\object_detection/ 

# Descripción del Sistema Propuesto

Para cumplir con el objetivo de lograr determinar si un conductor se ha comportado de manera imprudente o no. Se ha decidido utilizar una cámara colocada sobre el panel frontal del vehículo mirando hacia la misma dirección del movimiento. De esta manera se busca poder observar lo mismo que el conductor y así poder detectar si el conductor al pasar por alguna señal de tránsito importante realizó la acción pertinente. La evaluación de la acción se basará en dos entradas, la primera es el tipo de señal detectada y la segunda es el cambio de velocidad durante la detección de la señal. A partir de estos dos datos de entrada, el sistema permite inferir si el comportamiento durante un tramo de conducción fue correcto o no y muestra en pantalla mensajes visuales con su resultado. 

Por razones de dificultad de implementación, las pruebas se realizaron en simulación. Se utilizó el entorno de GTAV para entrenar al clasificador y para probarlo. Los datos de velocidad fueron simulados utilizando un joystick.

El sistema propuesto se basa en 4 bloques importantes que se ilustran en la Figura 1. En el primer bloque se encuentra la detección de señales que funciona como máster y da paso al resto de etapas. Una vez que se ha detectado la señal y ésta ha sido observada en 3 frames consecutivos, el algoritmo entrega una etiqueta con la clase de la señal detectada y empieza a guardar los datos de velocidad hasta que deje de detectar la señal. Durante la detección entra en funcionamiento el bloque de análisis de trama ya que el algoritmo de detección analiza una sola imagen pero el objectivo del sistema es analizar un comportamiento durante un tiempo de detección. Después, con el vector de velocidad guardado, se activa el bloque de análisis de velocidad que entrega como salida una etiqueta que indica el cambio de velocidad realizado durante la detección. A partir de las dos etiquetas entregadas se activa el último bloque de inferencia que, basado en una tabla de decisión, entrega una etiqueta de comportamiento bueno o malo y la presenta visualmente.

![Esquema Propuesto](/Imagenes/esquema.JPG)

## Detección de Señales 

El clasificador escogido para esta etapa fue una red convolucional del tipo Faster R-CNN. Este tipo de red basada en regiones de interes es una de las arquitecturas más populares para aplicaciones que requieren tanto velocidad de detección como exactitud. En comparación con la también muy conocida YOLO, la Faster R-CNN tiene un mejor desempeño cuando se desea detectar objectos muy pequeños. Por esta razón es la mejor opción para nuestra aplicación en el caso de querer detectar señales como semáforos a distancias muy alejadas.

Las señales o etiquetas con las que fue entrenado el clasificador fueron 8: 
* Semaforos en Rojo, Amarillo y Verde
* Señal vertical de Pare
* Señal vertical de Cruce Peatonal
* Señal de paso cebra pintada sobre el suelo
* Señal vertical de ceda el Paso
* Peatones

Para su entrenamiento se utilizaron 5600 imágenes en total. El 80% de las imágenes se utilizaron para training y el 20% para test. Se utilizó una red preentrenada de tipo Faster R-CNN que cuenta con dos modulos. El primer módulo es una red que entrega regiones de interes y el segundo modulo es un clasificador que utiliza las regiones entregadas por el modulo anterior. El estractor de caracteristicas utilizado fue el "Inception v2". La Figura 2. muestra el "Loss" histórico entregado por tensorflow durante su entrenamiento.

![Entrenamiento](/Imagenes/18.JPG)

El número de etiqueta por clase dentro del entranamiento y el test fueron:

* Labels Train:
Rojo: 4908, Amarillo: 2533, Verde: 2419, Ceda: 149, Cruce: 816, Cebra: 944, Pare: 545, Peatón: 1790
* Labels Test:
Rojo: 1427, Amarillo: 636, Verde: 695, Ceda: 78, Cruce: 233, Cebra: 187, Pare: 142, Peatón: 552

La Figura 3. muestra el resultado de ejecutar el script "Confusion_Matrix.py" con las imágenes de test. En la matriz resalta las equivocaciones de la red en detectar luz amarilla, esta casi siempre se confunde con luz verde. Esto debido a que el tipo de red no funciona bien para detectar objectos muy pequeños, y las imagenes con las que se testeo contienen varias tomas de semáforos a distancias muy alejadas. Es tambien por esta razón que muchas veces no se logra detectar los objetos, en consecuencia, la última fila de la matriz presenta falsos negativos.

![confusion_deteccion](/Imagenes/confusion_obj.jpg)

## Análisis de Velocidad
El vector con los datos de velocidad está determinado por el bloque de detección de señales. Esto significa que durante el análisis de la conducción, el tamaño del vector de velocidad no se mantendrá constante. En consecuencia, el algoritmo para el análisis de velocidad debe ser independiente del tamaño del vector. 

El algoritmo entrega como resultado una etiqueta que indica que sucedió con la velocidad durante la detección de la señal de tránsito, las etiquetas posibles son:
* Aumenta: 
* Disminuye
* Detiene
* Igual

El esquema propuesto para el algoritmo de velocidad se muestra en la Figura 4. Una vez que se reciben los datos de velocidad, se aplica un algoritmo de clustering para caracterizar toda la señal, se decidió trabajar con algoritmos que reciban como conocimiento a priori el número de clusters. El algoritmo agrupará la señal en 3 clusters. Con los datos agrupados, se aplica una regresión lineal sobre cada grupo y se obtienen las pendientes de cada uno. Se ha entrenado una máquina de soporte vectorial multiclase (SVM) que recibe como vector de características los valores de las pendientes y determina a cual de las cuatro clases mencionadas pertenece.

![Esquema Análisis de Velocidad](/Imagenes/SpeedA.PNG)

El SVM fue entrenado con valores de pendientes de señales adquiridas con un joystick. El dataset tuvo un tamaño de 150 señales de cada clase. El training set constituyó el 80% del dataset, mientras que el testing set el 20%.

Para la implementación de las etapas de clustering, regresión lineal y el SVM se empleó la libería "sklearn" de pyhton. Se decidió probar con tres tipos de clustering diferentes: K-Means, mini-batch Kmeans y hierarchical clustering (agglomerative) para evaluar el desempeño con cada uno. Dado que los datos no tienen alta dimensionalidad, los tres algoritmos funcionaron bien, sin embargo mini-batch KMeans presenta el menor tiempo de ejecución con resultados comparables a K-Means. Aglomerative clustering tuvo el mayor tiempo de ejecución de los tres.

En la Figura 5. se muestra el resultado de ejecutar el archivo "ConfussionMatrix.py" que muestra el desempeño del clasificador. 

![Matriz de confusión: análisis de velocidad](/Imagenes/MatrizconfusionSpeedA.PNG)

## Análisis de Tramas de Imágenes

La etapa de detección de imágenes no es suficiente para el análisis de comportamiento. Es necesario un análisis de toda una trama de detección. Para esto es necesario tener criterios para determinar en que momento empezar una trama y cuando terminarla. 

* Inicio:
Para rechazar los falsos positivos, se añadio un contador que determina la continuidad de la detección. Si una señal fue detectada 3 veces seguidas se asegura que la señal no fue un falso positivo. Además, se debe tomar en cuenta la ubicación de la señal detectada ya que un falso positivo puede aparecer en cualquier región de la escena, es así que se debe comparar siempre la ubicación del objeto previamente detectado con el que tenga la misma etiqueta en la siguiente escena y solo en ese caso contarlo.
* Intermedio:
En el caso de detectar más de una señal durante la trama, se añade prioridades a las distintas clases. Las clases con mayor prioridad son las luces de semáforo y pare y depues los cruces y ceda. De esta manera se determina una única detección al final de la trama, entregando como salida una única señal detectada.
* Final:
Para saber en que momento dejar de detectar y comenzar la etapa de análisis de velocidad se debe considerar que al dejar de detectar una señal aun puede aparecer otra con mayor prioridad (semáforo) a continuación por lo que se agrega un tiempo de espera. Si después de pasado un tiempo se dejo de detectar señales se procede a terminar la detección de esa trama.
Además se debe considerar que existen casos donde no se debe detectar una señal después de la detección de otra. El ejemplo más claro es la detección de una luz roja después de analizado el comportamiento de la luz amarilla.

## Inferencia del Comportamiento

En la última etapa del sistema, se utilizan las etiquetas entregadas por la etapa de análisis de tramas y el análisis de velocidad. Basandose en la siguiente tabla, se asigna una tarjeta roja o verde a la acción del conductor.

![tabla](/Imagenes/tabla.jpg)

# Resultados 

El siguiente enlace direcciona a un video demostrativo. En este video se presenta la detección de señales en diferentes escenarios (buena y mala conducción). En todos los casos se presenta una interfaz con mensajes sobre la pantalla que dependen de la señal detectada, la velocidad analizada y el comportamiento resultante.

https://youtu.be/5kdcxp4PEm0

A pesar de tener una existencia continua de falsos positivos, gracias al bloque de análisis de trama, se logra atenuar su influencia.
