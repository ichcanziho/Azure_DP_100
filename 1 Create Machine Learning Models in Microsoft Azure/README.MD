# Create Machine Learning Models in Microsoft Azure

# Índice

- [1 Explore data and create models to predict numeric values](#1-explore-data-and-create-models-to-predict-numeric-values)
  - [1.1 Videos](#11-videos)
  - [1.2 Readings](#12-readings)
  - [1.3 Quizzes](#13-quizzes)

- [2 Train and evaluate classification and clustering models](#2-train-and-evaluate-classification-and-clustering-models)
  - [2.1 Videos](#21-videos)
  - [2.2 Readings](#22-readings)
  - [2.3 Quizzes](#23-quizzes)

- [3 Train and evaluate deep learning models](#3-train-and-evaluate-deep-learning-models)
  - [3.1 Videos](#31-videos)
  - [3.2 Readings](#32-readings)
  - [3.3 Quizzes](#33-quizzes)

# 1 Explore data and create models to predict numeric values

Exploración y análisis de datos en el núcleo de la ciencia de datos. Los científicos de datos requieren habilidades en lenguajes como Python para explorar, visualizar y manipular datos. En este módulo, aprenderás a utilizar Python para explorar, visualizar y manipular datos. También aprenderás cómo la regresión se puede utilizar para crear un modelo de aprendizaje automático que predice valores numéricos. Utilizarás el framework scikit-learn en Python para entrenar y evaluar un modelo de regresión.

## 1.1 Videos

- [V1.1 Introduction to the course](#v11-introduction-to-the-course)
- [V1.2 Exam DP-100: Designing and Implementing a Data Science Solution on Azure certification](#v12-exam-dp-100-designing-and-implementing-a-data-science-solution-on-azure-certification)
- [V1.3 Lesson introduction](#v13-lesson-introduction)
- [V1.4 Lesson summary](#v14-lesson-summary)
- [V1.5 Lesson introduction](#v15-lesson-introduction)
- [V1.6 What is regression?](#v16-what-is-regression)
- [V1.7 Lesson summary](#v17-lesson-summary)

### V1.1 Introduction to the course

![1.png](ims%2FC1%2F1.png)

En este curso, aprenderás a crear modelos de aprendizaje automático en Microsoft Azure. El aprendizaje automático es fundamental para la modelización predictiva y la inteligencia artificial. El curso se compone de varios módulos:

![2.png](ims%2FC1%2F2.png)


![3.png](ims%2FC1%2F3.png)

**Módulo 1: Exploración de datos y creación de modelos para predecir valores numéricos**

- En este módulo, se enfoca en la exploración y análisis de datos, habilidades esenciales en la ciencia de datos.
- Se utiliza Python para explorar, visualizar y manipular datos.
- Se describe cómo se puede usar la regresión para predecir valores numéricos.
- Se emplea el framework Scikit-learn en Python para entrenar y evaluar un modelo de regresión.

![4.png](ims%2FC1%2F4.png)

**Módulo 2: Entrenamiento y evaluación de modelos de clasificación y agrupación**

- Se aborda la clasificación, una técnica de aprendizaje automático para categorizar elementos en clases.
- Se utiliza el framework Scikit-learn en Python para entrenar y evaluar un modelo de clasificación.
- Se explora cómo la agrupación se emplea para crear modelos de aprendizaje automático no supervisados que agrupan observaciones de datos en clusters.
- Se entrena un modelo de agrupación utilizando Scikit-learn en Python.

![5.png](ims%2FC1%2F5.png)

**Módulo 3: Entrenamiento y evaluación de modelos de aprendizaje profundo**

- Se enseñan los principios fundamentales del aprendizaje profundo y cómo crear modelos de redes neuronales profundas usando PyTorch o TensorFlow.
- Se explora el uso de redes neuronales convolucionales para crear modelos de clasificación de imágenes.
- A lo largo del curso, se ofrece la oportunidad de explorar modelos de aprendizaje automático en Microsoft Azure mediante ejercicios interactivos y ejemplos del mundo real de la ciencia de datos en acción, evaluaciones de conocimientos y exámenes prácticos.

![6.png](ims%2FC1%2F6.png)

Finalmente, al concluir cada sección habrá un apartado de ejercicios interactivos, repasos de información y exámenes prácticos.

### V1.2 Exam DP-100: Designing and Implementing a Data Science Solution on Azure certification

![7.png](ims%2FC1%2F7.png)

La certificación Microsoft Certified: Azure Data Scientist Associate es una oportunidad para demostrar el conocimiento en la aplicación de la ciencia de datos y el aprendizaje automático para implementar y ejecutar cargas de trabajo de aprendizaje automático en Azure. Esta certificación está diseñada para candidatos con experiencia en ciencia de datos y el uso de Azure Machine Learning y Azure Databricks. El examen de certificación generalmente consta de entre 40 y 60 preguntas, aunque el número puede variar según el examen. Se utilizan diferentes tipos de preguntas, como escenarios, respuestas cortas, opciones múltiples, revisión de marcas y arrastrar y soltar.

![8.png](ims%2FC1%2F8.png)

El examen abarca las siguientes áreas de dominio: gestionar recursos de Azure para el aprendizaje automático, ejecutar experimentos y entrenar modelos, implementar y operativizar soluciones de aprendizaje automático y aplicar principios éticos en el aprendizaje automático. Cuanto mayor sea el porcentaje, más preguntas contendrá esa sección específica del examen.

![9.png](ims%2FC1%2F9.png)

Para prepararte, tendrás acceso a la información necesaria a través de este programa. Una vez que completes con éxito un curso, recibirás un certificado de Coursera, y al finalizar el programa, recibirás un certificado adicional. Además, recibirás una notificación electrónica que te permitirá desbloquear un código de descuento del 50 por ciento para la certificación. 

![10.png](ims%2FC1%2F10.png)

**Ejemplos de candidatos y sus trayectorias:**

- **Carlos**, con experiencia en pruebas de software, decidió cambiar de carrera y completó con éxito la certificación Microsoft Azure Data Scientist Associate, lo que le ayudó a conseguir un empleo en el campo de la ciencia de datos.

- **June**, con una sólida experiencia en datos, aprovechó la certificación para avanzar en su carrera y abrir nuevas oportunidades en su campo.

- **Vikram**, un principiante, está trabajando en obtener la experiencia laboral necesaria para completar la certificación Microsoft Certified: Azure Data Scientist Associate y avanzar en su carrera.

### V1.3 Lesson introduction

![11.png](ims%2FC1%2F11.png)

En esta lección, aprenderás a explorar y analizar datos con Python. La función principal de los científicos de datos es, como era de esperar, explorar y analizar datos. Los resultados de un análisis pueden servir de base para un informe o un modelo de aprendizaje automático. Pero todo comienza con los datos, y Python es el lenguaje de programación más popular para los científicos de datos.

![12.png](ims%2FC1%2F12.png)

Después de décadas de desarrollo de código abierto, Python ofrece una amplia funcionalidad con potentes bibliotecas estadísticas y numéricas. NumPy y Pandas simplifican el análisis y la manipulación de datos. Matplotlib proporciona visualizaciones de datos atractivas. Scikit-learn ofrece un análisis de datos predictivo simple y efectivo. TensorFlow y PyTorch proporcionan capacidades de aprendizaje automático y aprendizaje profundo.

![13.png](ims%2FC1%2F13.png)

Por lo general, un proyecto de análisis de datos está diseñado para establecer ideas en torno a un escenario específico o para probar una hipótesis. Por ejemplo, supongamos que un profesor universitario recopila datos de sus estudiantes, incluyendo el número de clases a las que asisten, las horas que pasan estudiando y la calificación final obtenida en el examen de fin de curso.

![14.png](ims%2FC1%2F14.png)

El profesor podría analizar los datos para determinar si existe una relación entre la cantidad de estudio que realiza un estudiante y la calificación final que obtienen. El profesor podría usar los datos para probar una hipótesis de que solo los estudiantes que estudian durante un número mínimo de horas pueden esperar obtener una calificación aprobatoria.

![15.png](ims%2FC1%2F15.png)

![16.png](ims%2FC1%2F16.png)

En esta lección, aprenderás tareas comunes de exploración y análisis de datos, y explorarás cómo utilizar paquetes de Python como NumPy, Pandas y Matplotlib para analizar datos.

### V1.4 Lesson summary

### V1.5 Lesson introduction

### V1.6 What is regression?

### V1.7 Lesson summary

## 1.2 Readings

- [R1.1 Course syllabus](#r11-course-syllabus)
- [R1.2 Exam DP-100: Skills Measured](#r12-exam-dp-100-skills-measured)
- [R1.3 How to be successful in this course](#r13-how-to-be-successful-in-this-course)
- [R1.4 Explore data with NumPy and Pandas](#r14-explore-data-with-numpy-and-pandas)
- [R1.5 Exercise - Explore data with NumPy and Pandas](#r15-exercise-explore-data-with-numpy-and-pandas)
- [R1.6 Visualize data](#r16-visualize-data)
- [R1.7 Exercise - Visualize data with Matplotlib](#r17-exercise-visualize-data-with-matplotlib)
- [R1.8 Examine real world data](#r18-examine-real-world-data)
- [R1.9 Exercise - Examine real world data](#r19-exercise-examine-real-world-data)
- [R1.10 Exercise - Train and evaluate a regression model](#r110-exercise-train-and-evaluate-a-regression-model)
- [R1.11 Discover new regression models](#r111-discover-new-regression-models)
- [R1.12 Exercise - Experiment with more powerful regression models](#r112-exercise-experiment-with-more-powerful-regression-models)
- [R1.13 Improve models with hyperparameters](#r113-improve-models-with-hyperparameters)
- [R1.14 Exercise - Optimize and save models](#r114-exercise-optimize-and-save-models)

### R1.1 Course syllabus

El aprendizaje automático es la base de la modelización predictiva e inteligencia artificial. En este curso, aprenderás los principios fundamentales del aprendizaje automático y cómo utilizar herramientas y marcos comunes para entrenar, evaluar y utilizar modelos de aprendizaje automático.

**Módulo 1: Exploración de datos y creación de modelos para predecir valores numéricos**

La exploración y el análisis de datos son fundamentales en la ciencia de datos. Los científicos de datos necesitan habilidades en lenguajes como Python para explorar, visualizar y manipular datos. En este módulo, aprenderás a utilizar Python para explorar, visualizar y manipular datos. También aprenderás cómo la regresión se puede utilizar para crear un modelo de aprendizaje automático que predice valores numéricos. Utilizarás el framework scikit-learn en Python para entrenar y evaluar un modelo de regresión.

**Módulo 2: Entrenar y evaluar modelos de clasificación y agrupación**

La clasificación es un tipo de aprendizaje automático que se utiliza para categorizar elementos en clases. En este módulo, aprenderás cómo la clasificación se puede utilizar para crear un modelo de aprendizaje automático que predice categorías o clases. Utilizarás el framework scikit-learn en Python para entrenar y evaluar un modelo de clasificación. También aprenderás cómo la agrupación se puede utilizar para crear modelos de aprendizaje automático no supervisados que agrupan observaciones de datos en clústeres. Utilizarás el framework scikit-learn en Python para entrenar un modelo de agrupación.

**Módulo 3: Entrenar y evaluar modelos de aprendizaje profundo**

En este módulo, aprenderás los principios fundamentales del aprendizaje profundo y cómo crear modelos de redes neuronales profundas utilizando PyTorch o TensorFlow. También explorarás el uso de redes neuronales convolucionales para crear modelos de clasificación de imágenes.

### R1.2 Exam DP-100: Skills Measured

**Perfil del público objetivo**

Los candidatos para la certificación Azure Data Scientist Associate deben tener experiencia en la aplicación de la ciencia de datos y el aprendizaje automático para implementar y ejecutar cargas de trabajo de aprendizaje automático en Azure.

Las responsabilidades de este rol incluyen planificar y crear un entorno de trabajo adecuado para las cargas de trabajo de ciencia de datos en Azure. Realizas experimentos de datos y entrenas modelos predictivos. Además, gestionas, optimizas y despliegas modelos de aprendizaje automático en producción.

Un candidato para esta certificación debe tener conocimientos y experiencia en ciencia de datos y en el uso de Azure Machine Learning y Azure Databricks.

## Habilidades evaluadas

## Administrar recursos de Azure para el aprendizaje automático (25-30%)

**Crear un espacio de trabajo de Azure Machine Learning**

- Configurar ajustes del espacio de trabajo
- Gestionar un espacio de trabajo utilizando Azure Machine Learning Studio

**Gestionar datos en un espacio de trabajo de Azure Machine Learning**

- Seleccionar recursos de almacenamiento de Azure
- Registrar y mantener almacenes de datos
- Crear y gestionar conjuntos de datos

**Gestionar cómputo para experimentos en Azure Machine Learning**

- Determinar las especificaciones de cómputo apropiadas para una carga de trabajo de entrenamiento
- Crear objetivos de cómputo para experimentos y entrenamiento
- Configurar recursos de cómputo adjuntos, incluido Azure Databricks
- Monitorizar la utilización de cómputo

**Implementar seguridad y control de acceso en Azure Machine Learning**

- Determinar los requisitos de acceso y asignar requisitos a roles integrados
- Crear roles personalizados
- Gestionar la membresía de roles
- Gestionar credenciales utilizando Azure Key Vault

**Configurar un entorno de desarrollo de Azure Machine Learning**

- Crear instancias de cómputo
- Compartir instancias de cómputo
- Acceder a espacios de trabajo de Azure Machine Learning desde otros entornos de desarrollo

**Configurar un espacio de trabajo de Azure Databricks**

- Crear un espacio de trabajo de Azure Databricks
- Crear un clúster de Azure Databricks
- Crear y ejecutar cuadernos en Azure Databricks
- Vincular un espacio de trabajo de Azure Databricks a un espacio de trabajo de Azure Machine Learning

## Ejecutar experimentos y entrenar modelos (20-25%)

**Crear modelos utilizando el diseñador de Azure Machine Learning**

- Crear un pipeline de entrenamiento utilizando el diseñador de Azure Machine Learning
- Ingerir datos en un pipeline de diseñador
- Utilizar módulos de diseñador para definir un flujo de datos del pipeline
- Utilizar módulos de código personalizado en el diseñador

**Ejecutar scripts de entrenamiento de modelos**

- Crear y ejecutar un experimento utilizando el SDK de Azure Machine Learning
- Configurar ajustes de ejecución para un script
- Consumir datos de un conjunto de datos en un experimento utilizando el SDK de Azure Machine Learning
- Ejecutar un script de entrenamiento en Azure Databricks Compute
- Ejecutar código para entrenar un modelo en un cuaderno de Azure Databricks

**Generar métricas a partir de la ejecución de un experimento**

- Registrar métricas de la ejecución de un experimento
- Recuperar y ver salidas de experimentos
- Utilizar registros para solucionar problemas de errores de ejecución de experimentos
- Utilizar MLflow para rastrear experimentos
- Rastrear experimentos en ejecución en Azure Databricks

**Utilizar el Aprendizaje Automático Automatizado para crear modelos óptimos**

- Utilizar la interfaz de Aprendizaje Automático Automatizado en Azure Machine Learning Studio
- Utilizar el Aprendizaje Automático Automatizado desde el SDK de Azure Machine Learning
- Seleccionar opciones de preprocesamiento
- Seleccionar los algoritmos a buscar
- Definir una métrica principal
- Obtener datos para una ejecución de Aprendizaje Automático Automatizado
- Recuperar el mejor modelo

**Ajustar hiperparámetros con Azure Machine Learning**

- Seleccionar un método de muestreo
- Definir el espacio de búsqueda
- Definir la métrica principal
- Definir opciones de terminación anticipada
- Encontrar el modelo con valores óptimos de hiperparámetros

## Implementar y operativizar soluciones de aprendizaje automático (35-40%)

**Seleccionar cómputo para el despliegue de modelos**

- Considerar la seguridad para servicios desplegados
- Evaluar opciones de cómputo para el despliegue

**Desplegar un modelo como un servicio**

- Configurar ajustes de despliegue
- Desplegar un modelo registrado
- Desplegar un modelo entrenado en Azure Databricks en un punto de conexión de Azure Machine Learning
- Consumir un servicio desplegado
- Solucionar problemas de contenedor de despliegue

**Gestionar modelos en Azure Machine Learning**

- Registrar un modelo entrenado
- Supervisar el uso del modelo
- Supervisar el cambio de datos

**Crear un pipeline de Azure Machine Learning para inferencia por lotes**

- Configurar un ParallelRunStep
- Configurar cómputo para un pipeline de inferencia por lotes
- Publicar un pipeline de inferencia por lotes
- Ejecutar un pipeline de inferencia por lotes y obtener salidas
- Obtener salidas de un ParallelRunStep

**Publicar un pipeline de diseñador de Azure Machine Learning como un servicio web**

- Crear un recurso de cómputo de destino
- Configurar un pipeline de inferencia
- Consumir un punto de conexión desplegado

**Implementar prácticas de ML Ops**

- Desencadenar un pipeline de Azure Machine Learning desde Azure DevOps
- Automatizar la reentrenamiento del modelo basado en nuevas adiciones de datos o cambios en los datos
- Refactorizar cuadernos en scripts
- Implementar control de fuente para scripts

## Implementar aprendizaje automático responsable (5-10%)

**Utilizar intérpretes de modelos para interpretar modelos**

- Seleccionar un intérprete de modelo
- Generar datos de importancia de características

**Describir consideraciones de equidad para modelos**

- Evaluar la equidad del modelo basada en disparidad de predicción
- Mitigar la inequidad del modelo

**Describir consideraciones de privacidad para datos**

- Describir los principios de la privacidad diferencial
- Especificar niveles aceptables de ruido en los datos y los efectos en la privacidad

### R1.3 How to be successful in this course

## **Cómo tener éxito en este curso**

Tomar un curso en línea puede resultar abrumador. ¿Cómo aprendes a tu propio ritmo y logras con éxito tus objetivos?

Aquí tienes algunos consejos generales que pueden ayudarte a mantenerte enfocado y en el camino correcto:

### Establece metas diarias para estudiar:
Pregúntate qué esperas lograr en tu curso cada día. Establecer una meta clara puede ayudarte a mantenerte motivado y vencer la procrastinación. La meta debe ser específica y fácil de medir, como "Voy a ver todos los videos del Módulo 2 y completar la primera tarea de programación". ¡Y no olvides recompensarte cuando avances hacia tu objetivo!

### Crea un espacio de estudio dedicado:
Es más fácil recordar la información si te encuentras en el mismo lugar donde la aprendiste por primera vez. Por lo tanto, tener un espacio dedicado en casa para tomar cursos en línea puede hacer que tu aprendizaje sea más efectivo. Elimina cualquier distracción del espacio y, si es posible, mantenlo separado de tu cama o sofá. Una clara distinción entre donde estudias y donde tomas descansos puede ayudarte a concentrarte.

### Programa tiempo para estudiar en tu calendario:
Abre tu calendario y elige un momento predecible y confiable que puedas dedicar a ver clases y completar tareas. Esto ayuda a asegurarte de que tus cursos no se conviertan en la última tarea de tu lista de cosas por hacer.

> Consejo: Puedes agregar fechas límite para un curso de Coursera a tu calendario de Google, calendario de Apple u otra aplicación de calendario.

### Mantente responsable:
Cuéntales a tus amigos sobre los cursos que estás tomando, publica tus logros en tus cuentas de redes sociales o crea un blog sobre tus tareas. Tener una comunidad y una red de apoyo de amigos y familiares que te animen marca la diferencia.

### Toma notas de manera activa:
Tomar notas puede fomentar el pensamiento activo, aumentar la comprensión y ampliar tu capacidad de atención. Es una buena estrategia para interiorizar el conocimiento, ya sea que estés aprendiendo en línea o en el aula. Así que, toma una libreta o encuentra una aplicación digital que funcione mejor para ti y comienza a sintetizar puntos clave.

> Consejo: Mientras ves una conferencia en Coursera, puedes hacer clic en el botón "Guardar nota" debajo del video para guardar una captura de pantalla en tus notas del curso y agregar tus propios comentarios.

### Únete a la discusión:
Los foros de discusión del curso son un gran lugar para hacer preguntas sobre tareas, discutir temas, compartir recursos y hacer amigos. Nuestra investigación muestra que los estudiantes que participan en los foros de discusión son un 37% más propensos a completar un curso. ¡Así que publica hoy mismo!

### Haz una cosa a la vez:
Hacer varias cosas a la vez es menos productivo que centrarse en una tarea a la vez. Investigadores de la Universidad de Stanford encontraron que "las personas que reciben regularmente varias corrientes de información electrónica no pueden prestar atención, recordar información ni cambiar de una tarea a otra tan bien como aquellos que completan una tarea a la vez". Mantente enfocado en una tarea a la vez. Absorberás más información y completarás las tareas con mayor productividad y facilidad que si intentaras hacer muchas cosas al mismo tiempo.

### Tómate descansos:
Descansar tu mente después de aprender es fundamental para un alto rendimiento. Si te encuentras trabajando en un problema desafiante sin mucho progreso durante una hora, tómate un descanso. Salir a caminar, darte una ducha o hablar con un amigo puede revitalizarte e incluso darte nuevas ideas sobre cómo abordar ese proyecto.

-----

¡Tu viaje de aprendizaje de Microsoft Azure AI Fundamentals comienza ahora! Mientras te preparas para el examen AI-900 o trabajas para alcanzar tus metas de aprendizaje, te animamos a:

Revisar las pautas del examen y las habilidades evaluadas como punto de partida.
Trabajar en cada lección en la vía de aprendizaje. Intenta no saltarte ninguna actividad o lección a menos que estés seguro de que ya conoces esta información lo suficientemente bien como para avanzar.
Aprovechar la oportunidad de volver atrás y ver un video o leer la información adicional proporcionada antes de pasar a la siguiente lección o módulo.
Completar todos los cuestionarios, preguntas de práctica del examen y ejercicios. Durante las sesiones de práctica, tienes la oportunidad de repasar las preguntas nuevamente para asegurarte de estar satisfecho con tu progreso.
Leer cuidadosamente los comentarios al responder cuestionarios o exámenes de práctica, ya que esto te ayudará a reforzar lo que estás aprendiendo.
Utilizar el espacio de trabajo de Microsoft Azure Machine Learning. Para utilizar Azure Machine Learning, debes crear un espacio de trabajo en tu suscripción de Azure. Luego, puedes utilizar este espacio de trabajo para completar los ejercicios relacionados. Se te darán instrucciones claras sobre cómo configurarlo. Después de completar cada lección, también recibirás instrucciones sobre cómo "limpiar" para que puedas reutilizar el espacio

### R1.4 Explore data with NumPy and Pandas

Los científicos de datos pueden utilizar diversas herramientas y técnicas para explorar, visualizar y manipular datos. Una de las formas más comunes en las que los científicos de datos trabajan con datos es utilizando el lenguaje Python y algunos paquetes específicos para el procesamiento de datos.

#### ¿Qué es NumPy?
NumPy es una biblioteca de Python que proporciona funcionalidad comparable a herramientas matemáticas como MATLAB y R. Si bien NumPy simplifica significativamente la experiencia del usuario, también ofrece funciones matemáticas completas.

#### ¿Qué es Pandas?
Pandas es una biblioteca de Python extremadamente popular para el análisis y la manipulación de datos. Pandas es como Excel para Python, proporcionando funcionalidad fácil de usar para tablas de datos.
![17.png](ims%2FC1%2F17.png)

Tabla de datos con valores de nombre, edad, estado, número de hijos y mascotas.

#### Explora datos en un cuaderno Jupyter
Los cuadernos Jupyter son una forma popular de ejecutar scripts básicos utilizando tu navegador web. Normalmente, estos cuadernos son una sola página web, dividida en secciones de texto y secciones de código que se ejecutan en el servidor en lugar de en tu máquina local. Esto significa que puedes empezar rápidamente sin necesidad de instalar Python u otras herramientas.

#### Pruebas de hipótesis
La exploración y análisis de datos es típicamente un proceso iterativo, en el que el científico de datos toma una muestra de datos y realiza las siguientes tareas para analizarla y probar hipótesis:

- Limpia los datos para manejar errores, valores faltantes y otros problemas.
- Aplica técnicas estadísticas para comprender mejor los datos y cómo se espera que la muestra represente la población del mundo real de datos, teniendo en cuenta la variación aleatoria.
- Visualiza los datos para determinar las relaciones entre las variables y, en el caso de un proyecto de aprendizaje automático, identifica las características que podrían ser predictivas de la etiqueta.
- Revisa la hipótesis y repite el proceso.

### R1.5 Exercise - Explore data with NumPy and Pandas

> ## Nota:
> Puedes ver todo el notebook en formato HTML en [R1.5 Explore data with Numpy and Pandas.html](python_htmls%2FR1.5%20Explore%20data%20with%20Numpy%20and%20Pandas.html)

Explorando Datos con Python
Una parte significativa del rol de un científico de datos consiste en explorar, analizar y visualizar datos. Existe una amplia gama de herramientas y lenguajes de programación que pueden utilizar para hacer esto, y uno de los enfoques más populares es utilizar cuadernos Jupyter (como este) y Python.

Python es un lenguaje de programación flexible que se utiliza en una amplia variedad de escenarios, desde aplicaciones web hasta programación de dispositivos. Es extremadamente popular en la comunidad de ciencia de datos y aprendizaje automático debido a los numerosos paquetes que admite para el análisis y la visualización de datos.

En este ejercicio, explorarás arreglos de datos con NumPy y datos tabulares con Pandas.

Este ejercicio requiere el entorno de aprendizaje de Microsoft Learn para completarse. Puedes acceder al ejercicio desde el siguiente enlace.

https://learn.microsoft.com/en-us/training/modules/explore-analyze-data-with-python/3-exercise-explore-data

### R1.6 Visualize data

Los científicos de datos visualizan datos para comprenderlos mejor. Esto puede significar mirar los datos en bruto, medidas resumidas como promedios o representar los datos en gráficos. Los gráficos son una poderosa forma de ver los datos, ya que podemos discernir patrones moderadamente complejos rápidamente sin necesidad de definir medidas resumidas matemáticas.

**Representación visual de datos**

La representación visual de datos generalmente implica representarlos gráficamente. Esto se hace para proporcionar una evaluación cualitativa rápida de nuestros datos, lo que puede ser útil para comprender resultados, encontrar valores atípicos, comprender cómo se distribuyen los números, entre otros.

A veces sabemos de antemano qué tipo de gráfico será más útil, otras veces usamos gráficos de manera exploratoria. Para entender el poder de la visualización de datos, considera los datos a continuación: la ubicación (x, y) de un automóvil autónomo. En su forma cruda, es difícil ver patrones reales. La media o el promedio nos dice que su trayectoria se centró alrededor de x=0,2 e y=0,3, y el rango de números parece estar entre aproximadamente -2 y 2.

![19.png](ims%2FC1%2F19.png)

Si ahora representamos la ubicación-X en función del tiempo, podemos ver que aparentemente tenemos algunos valores faltantes entre los tiempos 7 y 12.

![20.png](ims%2FC1%2F20.png)

Coordenadas representadas a lo largo de un eje X que va de cero a veinte, y de dos a menos dos en el eje Y.

Si graficamos X vs Y, obtenemos un mapa de dónde ha conducido el automóvil. Es instantáneamente obvio que el automóvil ha estado conduciendo en círculo, pero en algún momento condujo al centro de ese círculo.

![21.png](ims%2FC1%2F21.png)

Coordenadas X e Y en un gráfico representado.

Los gráficos no se limitan a gráficos de dispersión en 2D como los anteriores, sino que se pueden utilizar para explorar otros tipos de datos, como proporciones, mostradas a través de gráficos circulares; gráficos de barras apiladas para mostrar cómo se distribuyen los datos, con histogramas, diagramas de caja y bigotes para mostrar cómo difieren dos conjuntos de datos.

A menudo, cuando intentamos comprender datos en bruto o resultados, podemos experimentar con diferentes tipos de gráficos hasta encontrar uno que explique los datos de una manera visualmente intuitiva.

### R1.7 Exercise - Visualize data with Matplotlib

> ## Nota:
> Puedes ver todo el notebook en formato HTML en [R1.7 Visualize data with Matplotlib.html](python_htmls%2FR1.7%20Visualize%20data%20with%20Matplotlib.html)

Los DataFrames proporcionan una excelente forma de explorar y analizar datos tabulares, pero a veces una imagen vale más que mil filas y columnas. La biblioteca Matplotlib proporciona la base para crear visualizaciones de datos que pueden mejorar en gran medida tu capacidad para analizar los datos.

Este ejercicio requiere el entorno de aprendizaje de Microsoft Learn para completarse. Puedes acceder al ejercicio desde el siguiente enlace.

https://learn.microsoft.com/en-us/training/modules/explore-analyze-data-with-python/5-exercise-visualize-data


### R1.8 Examine real world data

Los datos del mundo real pueden contener muchos problemas diferentes que pueden afectar la utilidad de los datos y nuestra interpretación de los resultados.

Es importante darse cuenta de que la mayoría de los datos del mundo real están influenciados por factores que no se registraron en ese momento. Por ejemplo, podríamos tener una tabla de tiempos de pistas de carreras de coches junto con tamaños de motor, pero es probable que también hayan influido otros factores que no se registraron, como el clima. Si estos factores problemáticos, la influencia de estos factores a menudo se puede reducir aumentando el tamaño del conjunto de datos.

En otras situaciones, los puntos de datos que claramente están fuera de lo esperado, también conocidos como 'valores atípicos', a veces se pueden eliminar de manera segura de los análisis, aunque se debe tener cuidado de no eliminar puntos de datos que proporcionen ideas reales.

Otro problema común en los datos del mundo real es el sesgo. El sesgo se refiere a la tendencia a seleccionar ciertos tipos de valores con más frecuencia que otros, de una manera que distorsiona la población subyacente o el 'mundo real'. A veces, el sesgo se puede identificar explorando los datos teniendo en cuenta el conocimiento básico sobre de dónde provienen los datos.

Recuerda, los datos del mundo real siempre tendrán problemas, pero este problema a menudo es superable. Recuerda:

- Verifica la presencia de valores faltantes y datos mal registrados.
- Considera la eliminación de valores atípicos obvios.
- Piensa en qué factores del mundo real podrían afectar tu análisis y considera si el tamaño de tu conjunto de datos es lo suficientemente grande para manejar esto.
- Verifica si los datos en bruto tienen sesgo y considera tus opciones para corregirlo, si se encuentra.

### R1.9 Exercise - Examine real world data

### R1.10 Exercise - Train and evaluate a regression model

### R1.11 Discover new regression models

### R1.12 Exercise - Experiment with more powerful regression models

### R1.13 Improve models with hyperparameters

### R1.14 Exercise - Optimize and save models

## 1.3 Quizzes

- [Q1.1 Test prep](#q11-test-prep)
- [Q1.2 Exercise Quiz 1](#q12-exercise-quiz-1)
- [Q1.3 Exercise Quiz 2](#q13-exercise-quiz-2)
- [Q1.4 Exercise Quiz 3](#q14-exercise-quiz-3)
- [Q1.5 Knowledge check 1](#q15-knowledge-check-1)
- [Q1.6 Exercise quiz 4](#q16-exercise-quiz-4)
- [Q1.7 Exercise quiz 5](#q17-exercise-quiz-5)
- [Q1.8 Exercise quiz 6](#q18-exercise-quiz-6)
- [Q1.9 Knowledge check 2](#q19-knowledge-check-2)

### Q1.1 Test prep

### Q1.2 Exercise Quiz 1

![18.png](ims%2FC1%2F18.png)

### Q1.3 Exercise Quiz 2

![22.png](ims%2FC1%2F22.png)

### Q1.4 Exercise Quiz 3

### Q1.5 Knowledge check 1

### Q1.6 Exercise quiz 4

### Q1.7 Exercise quiz 5

### Q1.8 Exercise quiz 6

### Q1.9 Knowledge check 2


# 2 Train and evaluate classification and clustering models

La clasificación es un tipo de aprendizaje automático utilizado para categorizar elementos en clases. En este módulo, aprenderás cómo la clasificación se puede utilizar para crear un modelo de aprendizaje automático que predice categorías o clases. Utilizarás el framework scikit-learn en Python para entrenar y evaluar un modelo de clasificación. También aprenderás cómo la agrupación se puede utilizar para crear modelos de aprendizaje automático no supervisados que agrupan observaciones de datos en clústeres. Utilizarás el framework scikit-learn en Python para entrenar un modelo de agrupación.

## 2.1 Videos

- [V2.1 Lesson introduction](#v21-lesson-introduction)
- [V2.2 What is classification?](#v22-what-is-classification)
- [V2.3 Evaluate classification models](#v23-evaluate-classification-models)
- [V2.4 Lesson summary](#v24-lesson-summary)
- [V2.5 Lesson introduction](#v25-lesson-introduction)
- [V2.6 What is clustering](#v26-what-is-clustering)
- [V2.7 Lesson summary](#v27-lesson-summary)

### V2.1 Lesson introduction

### V2.2 What is classification?

### V2.3 Evaluate classification models

### V2.4 Lesson summary

### V2.5 Lesson introduction

### V2.6 What is clustering

### V2.7 Lesson summary

## 2.2 Readings

- [R2.1 Exercise - Train and evaluate a classification model](#r21-exercise-train-and-evaluate-a-classification-model)
- [R2.2 Exercise - Perform classification with alternative metrics](#r22-exercise-perform-classification-with-alternative-metrics)
- [R2.3 Create multiclass classification models](#r23-create-multiclass-classification-models)
- [R2.4 Exercise - Train and evaluate multiclass classification models](#r24-exercise-train-and-evaluate-multiclass-classification-models)
- [R2.5 Exercise - Train and evaluate a clustering model](#r25-exercise-train-and-evaluate-a-clustering-model)
- [R2.6 Evaluate different types of clustering](#r26-evaluate-different-types-of-clustering)
- [R2.7 Exercise - Train and evaluate advanced clustering models](#r27-exercise-train-and-evaluate-advanced-clustering-models)

### R2.1 Exercise - Train and evaluate a classification model

### R2.2 Exercise - Perform classification with alternative metrics

### R2.3 Create multiclass classification models

### R2.4 Exercise - Train and evaluate multiclass classification models

### R2.5 Exercise - Train and evaluate a clustering model

### R2.6 Evaluate different types of clustering

### R2.7 Exercise - Train and evaluate advanced clustering models

## 2.3 Quizzes

- [Q2.1 Test prep](#q21-test-prep)
- [Q2.2 Exercise quiz 1](#q22-exercise-quiz-1)
- [Q2.3 Exercise quiz 2](#q23-exercise-quiz-2)
- [Q2.4 Exercise quiz 3](#q24-exercise-quiz-3)
- [Q2.5 Knowledge check 1](#q25-knowledge-check-1)
- [Q2.6 Exercise quiz 4](#q26-exercise-quiz-4)
- [Q2.7 Exercise quiz 5](#q27-exercise-quiz-5)
- [Q2.8 Knowledge check 2](#q28-knowledge-check-2)

### Q2.1 Test prep

### Q2.2 Exercise quiz 1

### Q2.3 Exercise quiz 2

### Q2.4 Exercise quiz 3

### Q2.5 Knowledge check 1

### Q2.6 Exercise quiz 4

### Q2.7 Exercise quiz 5

### Q2.8 Knowledge check 2

# 3 Train and evaluate deep learning models

En este módulo, aprenderás sobre los principios fundamentales del aprendizaje profundo y cómo crear modelos de redes neuronales profundas utilizando PyTorch o TensorFlow. También explorarás el uso de redes neuronales convolucionales para crear modelos de clasificación de imágenes.
## 3.1 Videos

- [V3.1 Lesson introduction](#v31-lesson-introduction)
- [V3.2 Deep neural network concepts](#v32-deep-neural-network-concepts)
- [V3.3 Lesson summary](#v33-lesson-summary)
- [V3.4 Lesson introduction](#v34-lesson-introduction)
- [V3.5 Layers in a convolutional neural network](#v35-layers-in-a-convolutional-neural-network)
- [V3.6 Transfer learning](#v36-transfer-learning)
- [V3.7 Lesson summary](#v37-lesson-summary)
- [V3.8 Course wrap up](#v38-course-wrap-up)

### V3.1 Lesson introduction

### V3.2 Deep neural network concepts

### V3.3 Lesson summary

### V3.4 Lesson introduction

### V3.5 Layers in a convolutional neural network

### V3.6 Transfer learning

### V3.7 Lesson summary

### V3.8 Course wrap up


## 3.2 Readings

- [R3.1 Exercise - Train a deep neural network](#r31-exercise-train-a-deep-neural-network)
- [R3.2 Exercise - Train a convolutional neural network](#r32-exercise-train-a-convolutional-neural-network)
- [R3.3 Exercise - Use transfer learning](#r33-exercise-use-transfer-learning)
- [R3.4 What to expect next](#r34-what-to-expect-next)

### R3.1 Exercise - Train a deep neural network

### R3.2 Exercise - Train a convolutional neural network

### R3.3 Exercise - Use transfer learning

### R3.4 What to expect next


## 3.3 Quizzes

- [Q3.1 Test prep](#q31-test-prep)
- [Q3.2 Exercise quiz 1](#q32-exercise-quiz-1)
- [Q3.3 Knowledge check 1](#q33-knowledge-check-1)
- [Q3.4 Exercise quiz 2](#q34-exercise-quiz-2)
- [Q3.5 Exercise quiz 3](#q35-exercise-quiz-3)
- [Q3.6 Knowledge check 2](#q36-knowledge-check-2)

### Q3.1 Test prep

### Q3.2 Exercise quiz 1

### Q3.3 Knowledge check 1

### Q3.4 Exercise quiz 2

### Q3.5 Exercise quiz 3

### Q3.6 Knowledge check 2

