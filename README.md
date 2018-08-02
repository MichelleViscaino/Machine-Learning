# Conductor Imprudente o no?
Este repositario contiene todos los archivos utilizados para crear un sistema que analiza de comportamiento de un conductor basado en detección de señales de tránsito utilizando Deep Learning. Lo que sigue es un pequeño tutorial para entender que arhivos son importantes y como utilizarlos. La columna vertebral del proyecto se basa en el API de detección de objectos de Tensor Flow para Windows. Este nos ayuda a entrenar un clasificador de multiples objetos utilizando herramientas de deep learning y posteriormente poder utilizarlo para aplicaciones en tiempo real. Es por esto que lo primero que se debe hacer es aprender a utilizar este API y después se procederá a explicar la aplicación que se le dió para la detección de señales de tránsito.

NOTA: A manera de ayuda, el siguiente enlace direcciona a un tutorial con un proyecto similar que explica muy detalladamente la manera de usar el API de detección, con énfasis en el entrenamiento.

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

# Como utilizar el Código

* Para que funcione el sistema se necesitan todas las carpetas y archivos que se generan por la descarga del API de detección. La carpeta "utils" contiene dos archivos importantes para nuestra aplicación. Estos archivos son "label_map_util" y "visualization_utils". El segundo archivo fue modificado para nuestra aplicación por lo que si se descarga la versión original del API no funcionará el sistema. La modificación se debe a que para la aplicación se requiere retornar variables de la detección que por default no son entregados por el API ya que este solo visualiza la detección, y el sistema requiere manejar los datos de ubicación y clase.

* La carpeta "inference_graph" es la carpeta que más espacio abarca ya que contiene el modelo del clasificador de objetos.

* El archivo "filename2.pkl" contiene el modelo del clasificador de velocidad.

* Se necesita crear la carpeta "test_images" que contiene las imagenes con las que se probará el sistema. Dentro de ella existen dos carpetas, una para guardar las imagenes que servirán como GroundTruth para determinar la matriz de confusión de la detección y la otra para probar todo el sistema durante una trama de conducción. La carpeta con el GroundTruth debe contener un archivo .csv con las etiquetas y la carpeta de prueba debe contener un archivo .xls con la trama de velocidad almacenada durante la conducción. 
* El script "Confusion_Matrix.py" sirve para determinar la matriz de confusión de la etapa de detección de señales.
* El script "Main.py" sirve para ejecutar el análisis de conducción. Este a su vez llama a varios otros scripts dentro de la carpeta "utils" y de este repositorio.

* Para poder ejecutar el código, es necesario tener instalado tensorflow y antes de ejecutarlo se debe asegurar de activar un ambiente de tensorflow y que asi se active la GPU con el comando: `C:\> activate tensorflow1`
Si se quiere ejecutar el código se debe considerar que el directorio por defecto con el que están configurados los programas es el siguiente: C:/> tensorflow1\models\research\object_detection/ 

# Descripción del Sistema Propuesto

Para cumplir con el objetivo de lograr determinar si un conductor se ha comportado de manera imprudente o no. Se ha decidido utilizar una cámara colocada sobre el panel frontal del vehículo mirando hacia la misma dirección del movimiento. De esta manera se busca poder observar los mismo que el conductor y asi poder detectar si el conductor al pasar por alguna señal de tránsito importante realizó la acción pertinente. La evaluación de la acción se basará en dos entradas, la primera es el tipo de señal detectada y la segunda es el cambio de velocidad durante la detección de la señal. A partir de estos dos datos de entrada, el sistema permite inferir si el comportamiento durante un tramo de conducción fue correcto o no y muestra en pantalla mensajes visuales con su resultado. 

Por razones de dificultad de implementación, las pruebas se realizaron en simulación. Se utilizó el entorno de GTAV para entrenar al clasificador y para probarlo. Los datos de velocidad fueron simulados utilizando un joystick.

El sistema propuesto se basa en 4 bloques importantes que se ilustran en la Figura 1. Primero se encuentra la detección de señales que funciona como master para dar el paso al resto de etapas. Una vez se ha detectado una señal se comienza a grabar la velocidad hasta que se la deje de detectar y sea entregada una etiqueta con la clase de señal detectada. Durante la detección entra en funcionamiento el bloque de análisis de trama ya que el algoritmo de detección analiza una sola imagen pero el objectivo del sistema es analizar un comportamiento durante un tiempo de detección. Después, cuando ya se tenga la trama de velocidad, se activa el bloque de análisis de velocidad que entrega como salida una etiqueta que indica el cambio de velocidad realizado durante la detección. A partir de las dos etiquetas entregadas se activa el último bloque de inferencia que basado en una tabla de decisión, entrega una etiqueta de comportamiento bueno o malo y la presenta visualmente.

![Esquema Propuesto](/Imagenes/esquema.JPG)

## Detección de Señales 

El clasificador escogido para esta etapa fue una red convolucional del tipo Faster R-CNN. Este tipo de red basada en regiones de interes es una de las arquitecturas más populares para aplicaciones que requieren tanto velocidad de detección como exactitud. En comparación con la también muy conocida YOLO, la Faster R-CNN tiene un mejor desempeño cuando se desea detectar objectos muy pequeños. Por esta razón es la mejor opción para nuestra aplicación en el caso de querer detectar señales como semáforos a distancias muy alejadas.

El entrenamiento se lo

## Análisis de Velocidad

## Análisis de Tramas de Imágenes

## Inferencia del Comportamiento

# Resultados 

