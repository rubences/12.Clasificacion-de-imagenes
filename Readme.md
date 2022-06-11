## Caso práctico alrededor de la moda
1. Presentación de Kaggle
Kaggle (www.kaggle.com) es el sitio de referencia en términos de proposiciones de desafíos relacionados con Machine Learning. Este sitio está repleto de casos de estudio y de retos propuestos por grandes grupos, como McDonald’s o Netflix. Todos podemos intentar resolver los distintos desafíos proponiendo nuestros propios modelos de predicción. Este sitio también permite aprender mucho sobre las técnicas relacionadas con Machine Learning porque todas las propuestas de solución enviadas por los participantes a los distintos desafíos son visibles, comentadas y así se pueden estudiar. Este sitio es completamente gratuito, no hay motivos para no visitarlo.
Vamos a realizar nuestro caso práctico a partir de este sitio. Este caso nos permitirá ilustrar los principios de programación de una red neuronal convolucional.
2. Módulo Keras
En el capítulo anterior utilizamos la biblioteca TensorFlow. Hay un módulo complementario a esta biblioteca llamado Keras que, ante todo, es fácil de usar. Esta facilidad reside especialmente en la fase de creación y de parametrización de los distintos modelos de aprendizaje, sin olvidar que también es muy adecuado para el uso de redes neuronales convolucionales porque posee las funciones adecuadas para ello de manera nativa.
3. Clasificar vestidos, jerséis y zapatos
Uno de los casos de estudio más utilizados en la puesta en práctica de las redes neuronales convolucionales es la clasificación de cartas manuscritas con ayuda de la colección de observaciones MNIST (Mixed National Institute of Standards and Technology). Sin embargo, en 2018 Zalando publicó su propia colección de imágenes bajo el nombre Zalando-MNIST. El propósito de esta colección de imágenes era permitir a los algoritmos entrenarse en la clasificación de objetos de moda: jersey, vestido, bolso, etc. Ahora vamos a utilizar está colección para nuestro caso práctico.
Más allá del aspecto, muy concreto, del proyecto, la mayoría de los ordenadores pueden ejecutar la fase de aprendizaje de este caso de estudio. En efecto, como ya hemos mencionado al principio de este capítulo, la clasificación de las imágenes requiere características adquiridas en términos de memoria, potencia de procesador y rendimiento de la tarjeta gráfica para obtener buenos resultados de manera rápida.
También es posible utilizar la nube alquilando máquinas virtuales situadas en servidores dedicados a la inteligencia artificial, como ofrece Amazon. Estas máquinas se pueden configurar en términos de potencia en función del proyecto que se desea realizar.
Ahora le invitamos a descargar los datos disponibles en el sitio del editor y después a crear un proyecto nuevo de Python en el que será necesario crear un directorio datas para depositar el conjunto de los datos que hemos descargado previamente.
 
Copia de los datos de aprendizaje en el directorio datas del proyecto
4. ¿De qué datos disponemos?
Antes de sumergirnos completamente en la fase de aprendizaje, tenemos que conocer los datos de los que disponemos.
Como queremos clasificar imágenes, deberíamos estar en posesión de algunas para realizar el aprendizaje. Pero está claro que no tenemos archivos de imagen como los que estamos acostumbrados a encontrar. En efecto, no hay archivos JPEG ni PNG.
Sin embargo, si abrimos el archivo fashion-mnist_train.csv, podemos darnos cuenta de que contiene observaciones con las siguientes características:
+	Una etiqueta.
+	Píxeles numerados del 1 al 784 con distintos valores.
Ahora veamos qué nos dice la documentación de este conjunto de observaciones disponible en el sitio de Kaggle (https://www.kaggle.com/plarmuseau/zalando-image-classifier/data):
+	El conjunto de observaciones contiene 60 000 imágenes de aprendizaje y 10 000 imágenes de prueba.
+	Todas las imágenes tienen una altura de 28 píxeles y una anchura de 28 píxeles, con un total de 784 píxeles.
+	Cada píxel está asociado a un solo valor de píxel que indica el brillo de este píxel. Este valor de píxel es un número entero comprendido entre 0 y 255.
+	Los conjuntos de datos de entrenamiento y de pruebas contienen 785 columnas. 
+	La primera columna está formada por etiquetas de clase y representa el artículo de vestir. El resto de las columnas contienen los valores de los píxeles de la imagen asociada.
Siempre a partir de la documentación, podemos determinar las distintas etiquetas:
+	0 - Camiseta/top
+	1 - Pantalón
+	2 - Jersey
+	3 - Vestido
+	4 - Abrigo
+	5 - Sandalias
+	6 - Camisa
+	7 - Zapatillas
+	8 - Bolso
+	9 - Botines

A partir de esta información podemos afirmar que nuestras imágenes tan esperadas están presentes en este archivo bajo la forma de valores de píxeles.