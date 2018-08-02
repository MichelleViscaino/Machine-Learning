# Conductor Imprudente o no?
Este repositario contiene todos los archivos utilizados para crear un sistema que analiza de comportamiento de un conductor basado en detección de señales de tránsito utilizando Deep Learning. Lo que sigue es un pequeño tutorial para entender que arhivos son importantes y como utilizarlos. La columna vertebral del proyecto se basa en el API de detección de objectos de Tensor Flow para Windows. Este nos ayuda a entrenar un clasificador de multiples objetos utilizando herramientas de deep learning y posteriormente poder utilizarlo para aplicaciones en tiempo real. Es por esto que lo primero que se debe hacer es aprender a utilizar este API y después se procederá a explicar la aplicación que se le dió para la detección de señales de tránsito.

NOTA: A manera de ayuda, el siguiente enlace direcciona a un tutorial con un proyecto similar que explica muy detalladamente la manera de usar el API de detección, con énfasis en el entrenamiento.

https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10

# Como utilizar el Código

* Las siguientes "data", "data_decoders", "dataset_tools", "doc", "g3doc","inference","matchers", "meta_architectures", "metrics", "models", "protos", "samples", "test_data", "utils" son carpetas propias del API y su uso será transperante para nosotros, exceptuando la última carpeta "utils" que contiene dos archivos importantes para nuestra aplicación. Estos archivos son "label_map_util" y "visualization_utils". El segundo archivo fue modificado para nuestra aplicación por lo que si se descarga la versión original del API no funcionará el sistema. La modificación se debe a que para la aplicación se requiere retornar variables de la detección que por default no son entregados por el API ya que este solo visualiza la detección, y el sistema requiere manejar los datos de ubicación y clase.

* La carpeta "inference_graph" es la carpeta que más espacio abarca ya que contiene el modelo del clasificador entrenado.

* La carpeta "test_images" contiene las imagenes con las que se probará el sistema. Dentro de ella existen dos carpetas, una para guardar las imagenes que servirán como GroundTruth para determinar la matriz de confusión de la detección y la otra para probar todo el sistema durante una trama de conducción. La carpeta con el GroundTruth debe contener imagenes etiquetadas y almacenadas en un archivo .csv. 
* El script "Confusion_Matrix.py" sirve para determinar la matriz de confusión de la etapa de detección de señales.
* El scripts "Main.py" sirve para ejecutar el análisis de conducción. Este a su vez llama a varios otros scripts dentro de la carpeta "utils" y fuera de las carpetas.

* Para poder ejecutar el código, es necesario tener instalado tensorflow y antes de ejecutarlo se debe asegurar de activar un ambiente de tensorflow y que asi se active la GPU con el comando: `C:\> activate tensorflow1`
Si se quiere ejecutar el código se debe considerar que el directorio por defecto con el que están configurados los programas es el siguiente: C:/> tensorflow1\models\research\object_detection/ 

## Training



# Descripción del Sistema Propuesto

## Detección de Señales 

## Análisis de Velocidad

## Análisis de Tramas de Imágenes

## Inferencia del Comportamiento

# Resultados 

