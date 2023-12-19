# MachineLearning

Con un sólido conocimiento en matemáticas, estadísticas y programación, he liderado proyectos que abarcan desde la clasificación y predicción hasta la generación de insights a partir de grandes conjuntos de datos. Mi enfoque se basa en comprender las necesidades del negocio y traducirlas en modelos predictivos y sistemas inteligentes que aporten un valor significativo.

Esta descripción presenta una visión general de las habilidades, experiencia y proyectos relevantes de un ingeniero de machine learning, destacando su enfoque en la resolución de problemas y su capacidad para generar valor a través de la aplicación de técnicas avanzadas de aprendizaje automático.

---
---

## Métodos Numéricos, Rusty Bargain

**Rusty Bargain** es un servicio de venta de coches de segunda mano que está desarrollando una app para atraer a nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial, especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.

A Rusty Bargain le interesa:
- la calidad de la predicción
- la velocidad de la predicción
- el tiempo requerido para el entrenamiento

**Instrucciones del Proyecto**

- Descargar y examinar los datos.
- Entrenar diferentes modelos con varios hiperparámetros (debemos hacer al menos dos modelos diferentes, pero más es mejor. Hacer varias implementaciones de potenciación del gradiente no cuentan como modelos diferentes). El punto principal de este paso es comparar métodos de potenciación del gradiente con bosque aleatorio, árbol de decisión y regresión lineal.
- Analiza la velocidad y la calidad de los modelos.

**Observaciones:**

- Utiliza la métrica RECM para evaluar los modelos.
- La regresión lineal no es muy buena para el ajuste de hiperparámetros, pero es perfecta para hacer una prueba de cordura de otros métodos. Si la potenciación del gradiente funciona peor que la regresión lineal, definitivamente algo salió mal.
- Aprender por propia cuenta sobre la librería LightGBM y sus herramientas para crear modelos de potenciación del gradiente (gradient boosting).
- Idealmente, el proyecto debe tener regresión lineal para una prueba de cordura, un algoritmo basado en árbol con ajuste de hiperparámetros (preferiblemente, bosque aleatorio), LightGBM con ajuste de hiperparámetros (probar un par de conjuntos), y CatBoost y XGBoost con ajuste de hiperparámetros (opcional).
- Tomar nota de la codificación de características categóricas para algoritmos simples. LightGBM y CatBoost tienen su implementación, pero XGBoost requiere OHE.
- Dado que el entrenamiento de un modelo de potenciación del gradiente puede llevar mucho tiempo, cambiaremos solo algunos parámetros del modelo.

  [Ir al Proyecto](https://github.com/juliocmi/MachineLearning/blob/main/ML_Projects/Me%CC%81todos%20Nume%CC%81ricos.ipynb)

---
---

## Predicciones con Álbebra Líneal, Sure Tomorrow

La compañía de seguros **Sure Tomorrow** quiere resolver varias tareas con la ayuda de machine learning y nos pide que evaluemos esa posibilidad.

- **Tarea 1:** encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
- **Tarea 2:** predecir si es probable que un nuevo cliente reciba un beneficio de seguro. ¿Puede un modelo de predicción funcionar mejor que un modelo ficticio?
- **Tarea 3:** predecir la cantidad de beneficios de seguro que probablemente recibirá un nuevo cliente utilizando un modelo de regresión lineal.
- **Tarea 4:** proteger los datos personales de los clientes sin romper el modelo de la tarea anterior.

Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento de datos u ofuscación de datos. Pero los datos deben protegerse de tal manera que la calidad de los modelos de machine learning no se vea afectada. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.

**Instrucciones del Proyecto**

- Carga los datos.
- Verifica que los datos no tengan problemas: no faltan datos, no hay valores extremos, etc.
- Trabaja en cada tarea y responde las preguntas planteadas en la plantilla del proyecto.
- Saca conclusiones basadas en tu experiencia trabajando en el proyecto.

[Ir al Proyecto](https://github.com/juliocmi/MachineLearning/blob/main/ML_Projects/Zyfra_ML_Project.ipynb)

---
---

## Predicción de la Cantidad de Oro, Zyfra Corporation

Ha llegado la hora de abordar un problema real de ciencia de datos que proviene del ámbito de la minería del oro. Este proyecto fue proporcionado por Zyfra Corporation.

Debemos preparar un prototipo de un modelo de machine learning para Zyfra. La empresa desarrolla soluciones de eficiencia para la industria pesada.
El modelo debe predecir la cantidad de oro extraído del mineral de oro. Disponemos de los datos de extracción y purificación.
El modelo ayudará a optimizar la producción y a eliminar los parámetros no rentables.

Tendremos que:

- Preparar los datos;
- Realizar el análisis de datos;
- Desarrollar un modelo y entrenarlo.

[Ir al Proyecto](https://github.com/juliocmi/MachineLearning/blob/main/ML_Projects/Zyfra_ML_Project.ipynb)
