#!/usr/bin/env python
# coding: utf-8

# # Zyfra Machine Learning Project
# 
# ---
# 
# *Fecha de Creación: Marzo 2023*
# 
# - **Senior Data Science:** Francisco Alfaro
# - **Instructor:** Alfonso Tobar
# - **Code Reviewer:** Marcos Torres
# - **NoteBook by:** Julio César Martínez
# 
# # Tabla de Contenido
# 
# ---
# 
# 1. Introducción.
# 2. Licencia.
# 3. Librerías Requeridas.
# 4. Definición del Problema.
# 5. Workflow.
# 6. Etapa Uno: Preparación de los Datos.
# 7. Etapa Dos: Análisis de Datos.
# 8. Etapa Tres: Construyendo el Modelo.
# 9. Conclusiones.
# 10. Bibliografía.
# 11. Agradecimientos.

# ## || Introducción.
# 
# ---
# 
# El oro se conoce desde la prehistoria. En el Antiguo Egipto, el faraón Dyer (ca. 3000 a. C.), llevaba en su título un jeroglífico referente al metal, y también se menciona varias veces en el Antiguo Testamento. Se ha considerado uno de los metales más preciosos a lo largo de la Historia, y como "valor patrón" se ha empleado profusamente, acuñado en monedas.
# 
# La mayor fuente de oro de la historia ha sido la cuenca de Witwatersrand en Sudáfrica. Witwatersrand representa aproximadamente el 30% de todo el oro extraído. Otras fuentes importantes de oro incluyen la mina Mponeng, extremadamente profunda, en Sudáfrica; las minas Super Pit y Newmont Boddington, en Australia; la mina Grasberg de Indonesia y las minas en el estado de Nevada, EE. UU.
# 
# El volumen de las reservas de oro se puede calcular con mayor precisión que los recursos, aunque todavía no es una tarea fácil. El stock subterráneo de reservas de oro se estima actualmente en alrededor de 50.000 toneladas, según el Servicio Geológico de Estados Unidos.
# 
# Para poner esto en perspectiva, ya se han extraído alrededor de 190.000 toneladas de oro en total, aunque las estimaciones varían. Esto significa, en base a estas cifras aproximadas, que todavía queda alrededor del 20% por explotar. En 2021, China fue el principal país productor de oro del mundo. Con aproximadamente 370 toneladas métricas de este metal precioso, el gigante asiático quedó por delante de otros países de grandes dimensiones como Australia y Rusia.
# 
# El oro puede encontrarse en la naturaleza en los ríos. Algunas piedras de los ríos contienen pepitas de oro en su interior. La fuerza del agua separa las pepitas de la roca y las divide en partículas minúsculas que se depositan en el fondo del cauce.
# 
# Los buscadores de oro localizan estas partículas de oro de los ríos mediante la técnica del bateo. El utensilio utilizado es la batea, un recipiente con forma de sartén. La batea se llena con arena y agua del río y se va moviendo provocando que los materiales de mayor peso, como el oro, sean depositados en el fondo y la arena superficial se desprenda.
# 
# El método de ensayo a fuego consiste en producir una fusión de la muestra usando reactivos fundentes adecuados para obtener dos fases líquidas: una escoria constituida principalmente por silicatos complejos y una fase metálica constituida por plomo, el cual colecta los metales nobles de interés (Au y Ag). Los dos líquidos se separan en dos fases debido a su respectiva inmiscibilidad y gran diferencia de densidad, éstos solidifican al enfriar. El plomo sólido (con los metales nobles colectados) es separado de la escoria como un régulo. Este régulo de plomo obtenido es oxidado en caliente en copela de magnesita y absorbido por ella, quedando en su superficie el botón de oro y plata, elementos que se determinan posteriormente por método gravimétrico (por peso) o mediante espectroscopia de Absorción Atómica.
# 
# Esta última parte del proceso para obtener oro es el objetivo de este proyecto.

# ## || Licencia.
# ---
# 
# Este notebook fue creado para la practica profesional de habilidades en ciencia de adtos y es propiedad de su creador. Queda prohibida su venta, copia, distribución, modificación y/o cualquier uso indevido e ilegial de este así como la base de datos (dataset) proporcionada por Zyfra. Si existe alguna duda o aclaración con respecto a este ejercicio puedes ponerte en contacto con el creador para dar solución.

# ## || Librerías Requeridas.
# 
# ---
# 
# Para este proyecto en este nootebook utilizamos diferentes librerías de python como son:
# 
# - Pandas.
# - NumPy.
# - Scikit-Learn.
# - Matplotlib.

# ## || Definición del Problema
# 
# ---
# 
# La empresa **Zyfra** desarrolla soluciones de eficiencia para la industria pesada. Para este proyecto **Zyfra** quiere preparar un prototipo de un modelo de machine learning. El modelo ayudará a optimizar la producción y a eliminar los parámetros no rentables para la extracción de oro.
# 
# Tenemos a disposición los datos en bruto que solamente fueron descargados del almacén de datos o **datawarehouse** antes de construir el modelo. Algunos parámetros no están disponibles porque fueron medidos o calculados mucho más tarde, por eso, algunas de las características que están presentes en el conjunto de entrenamiento pueden estar ausentes en el conjunto de prueba. El conjunto de prueba tampoco contiene objetivos, el dataset fuente contiene los conjuntos de entrenamiento y prueba con todas las características.
# 
# ### Instrucciones
# 
# - Preparar un prototipo de un modelo de machine learning para Zyfra. La empresa desarrolla soluciones de eficiencia para la industria pesada.
# - El modelo debe predecir la cantidad de oro extraído del mineral de oro. Disponemos de los datos de extracción y purificación.
# - El modelo ayudará a optimizar la producción y a eliminar los parámetros no rentables.
# 
# Tendremos que:
# 
# - preparar los datos;
# - realizar el análisis de datos;
# - desarrollar un modelo y entrenarlo.
# 
# Para completar el proyecto, podemos utilizar la documentación de pandas, matplotlib y sklearn.
# La siguiente lección trata sobre el proceso de depuración del mineral. Tocará seleccionar la información importante para el desarrollo del modelo.

# ## || Workflow
# 
# ---
# 
# ### ¿CÓMO SE EXTRAE EL ORO DEL MINERAL?
# 
# Veamos las etapas de este proceso.
# 
# El mineral extraído se somete a un tratamiento primario para obtener la mezcla de mineral, o alimentación rougher, que es la materia prima utilizada para la flotación (también conocida como proceso rougher). Después de la flotación, el material se somete al proceso de purificación en dos etapas.
# 
# ![img1](https://practicum-content.s3.us-west-1.amazonaws.com/new-markets/DS_sprint_10/ES/moved_10.3.2ES.png)
# 
# Veamos el proceso paso a paso:
# 
# **1. FLOTACIÓN**
# 
# La mezcla de mineral de oro se introduce en las plantas de flotación para obtener un concentrado de oro rougher y colas rougher (es decir, residuos del producto con una baja concentración de metales valiosos).
# La estabilidad de este proceso se ve afectada por la volatilidad y el estado físico-químico desfavorable de la pulpa de flotación (una mezcla de partículas sólidas y líquido).
# 
# **2. PURIFICACIÓN**
# 
# El concentrado rougher se somete a dos etapas de purificación. Luego de esto, tenemos el concentrado final y las nuevas colas.
# 
# *Descripción de datos*
# 
# **PROCESO TECNOLÓGICO**
# 
# - Rougher feed — materia prima
# - Rougher additions (o adiciones de reactivos) - reactivos de flotación: xantato, sulfato, depresante
# - Xantato — promotor o activador de la flotación
# - Sulfato — sulfuro de sodio para este proceso en particular
# - Depresante — silicato de sodio
# - Rougher process — flotación
# - Rougher tails — residuos del producto
# - Float banks — instalación de flotación
# - Cleaner process — purificación
# - Rougher Au — concentrado de oro rougher
# - Final Au — concentrado de oro final
# - Parámetros de las etapas
# - air amount — volumen de aire
# - fluid levels
# - feed size — tamaño de las partículas de la alimentación
# - feed rate
# 
# **DENOMINACIÓN DE LAS CARACTERÍSTICAS**
# 
# Así es como se denominan las características:
# `[stage].[parameter_type].[parameter_name]`
# 
# Ejemplo: rougher.input.feed_ag
# 
# Valores posibles para `[stage]`:
# 
# - rougher — flotación
# - primary_cleaner — purificación primaria
# - secondary_cleaner — purificación secundaria
# - final — características finales
# 
# Valores posibles para `[parameter_type]`:
# 
# - input — parámetros de la materia prima
# - output — parámetros del producto
# - state — parámetros que caracterizan el estado actual de la etapa
# - calculation — características de cálculo
# 
# 
# ![img2](https://practicum-content.s3.us-west-1.amazonaws.com/new-markets/DS_sprint_10/ES/moved_10.3.2.2ES.png)
# 
# **CÁLCULO DE LA RECUPERACIÓN**
# 
# Tienes que simular el proceso de recuperación del oro del mineral de oro.
# Utiliza la siguiente fórmula para simular el proceso de recuperación:
# 
# $$Recuperación = \frac{Cx(F-T)}{Fx(C-T)}x100$$
# 
# dónde:
# - C — proporción de oro en el concentrado justo después de la flotación (para saber la recuperación del concentrado rougher)/después de la purificación (para saber la recuperación del concentrado final)
# - F — la proporción de oro en la alimentación antes de la flotación (para saber la recuperación del concentrado rougher)/en el concentrado justo después de la flotación (para saber la recuperación del concentrado final)
# - T — la proporción de oro en las colas rougher justo después de la flotación (para saber la recuperación del concentrado rougher)/después de la purificación (para saber la recuperación del concentrado final)
# 
# Para predecir el coeficiente, hay que encontrar la proporción de oro en el concentrado y en las colas. Ten en cuenta que tanto el concentrado final como el concentrado rougher tienen importancia.
# 
# ---
# 
# **MÉTRICAS DE EVALUACIÓN**
# 
# Para resolver el problema, necesitaremos una nueva métrica. Se llama sMAPE, o error medio absoluto porcentual simétrico.
# 
# Es similar al MAE, pero se expresa en valores relativos en lugar de absolutos. ¿Por qué es simétrico? Porque tiene en cuenta la escala tanto del objetivo como de la predicción.
# 
# Así es como se calcula el sMAPE:
# 
# $$sMAPE = \frac{1}{N}\sum_{i=1}^{n}\frac{|yi - \widetilde{y}i|}{(|yi|+|\widetilde{y}i|)/2}$$
# 
# Designación:
# 
# - $y$ - Valor del objetivo para la observación con el índice i en el conjunto utilizado para medir la calidad.
# - $\widetilde{y}$ - Valor de la predicción para la observación con el índice i, por ejemplo, en la muestra de prueba.
# - $N$ - Número de observaciones de la muestra.
# - $\sum_{i=1}^{n}$ - Suma de todas las observaciones de la muestra (i toma valores de 1 a N).
# 
# Necesitamos predecir dos valores:
# 
# - La recuperación del concentrado rougher `rougher.output.recovery`.
# - La recuperación final del concentrado `final.output.recovery`.
# 
# La métrica final incluye los dos valores:
# 
# $$sMAPE Final = 25\% x sMAPE(rougher) + 75\% x sMAPE(final)$$

# ## || Etapa Uno: Preparación de los Datos
# 
# ---
# 
# Vamos a explorar los datos.

# In[32]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble        import RandomForestRegressor
from sklearn.tree            import DecisionTreeRegressor


pd.options.display.max_rows = 100


# Importar Datos.

# In[2]:


zyfra_train = pd.read_csv('/datasets/gold_recovery_train.csv', parse_dates = ["date"], index_col="date")
zyfra_test  = pd.read_csv('/datasets/gold_recovery_test.csv', parse_dates = ["date"], index_col="date")
zyfra_full  = pd.read_csv('/datasets/gold_recovery_full.csv', parse_dates = ["date"], index_col="date")


# Tamaño de los Sets

# In[3]:


print('Tamaño del 1er Dataset:', zyfra_train.shape)
print('Tamaño del 2do Dataset:', zyfra_test.shape)
print('Tamaño del 3er Dataset:', zyfra_full.shape)


# Tipo de Datos.

# In[4]:


print('Dataset 1:', zyfra_train.dtypes)
print('\n')
print('Dataset 2:', zyfra_test.dtypes)
print('\n')
print('Dataset 3:', zyfra_full.dtypes)


# Información General.

# In[5]:


print('Dataset 1:', zyfra_train.info())
print('\n')
print('Dataset 2:', zyfra_test.info())
print('\n')
print('Dataset 3:', zyfra_full.info())


# Visualizamos Sets.

# In[6]:


zyfra_train.head()


# In[7]:


zyfra_test.head()


# In[8]:


zyfra_full.head()


# **Observaciones**: 
# 
# > Se observan filas incompletas en el conjunto de prueba respecto al de entrenamiento. También hay presencia de valores ausentes. Antes de proceder al tratamiento de estos se nos ha pedido calcular el índice de recuperación.

# **Descripción de Datos** 
# 
# - `Rougher feed` — materia prima
# - `Rougher additions` (o adiciones de reactivos) - reactivos de flotación: xantato, sulfato, depresante
# - - Xantato — promotor o activador de la flotación
# - - Sulfato — sulfuro de sodio para este proceso en particular
# - - Depresante — silicato de sodio
# - `Rougher process` — flotación
# - `Rougher tails` — residuos del producto
# - `Float banks` — instalación de flotación
# - `Cleaner process` — purificación
# - `Rougher Au` — concentrado de oro rougher
# - `Final Au` — concentrado de oro final
# 
# Parámetros de las etapas
# 
# - `air amount` — volumen de aire
# - `fluid levels`
# - `feed size` — tamaño de las partículas de la alimentación
# - `feed rate`
# 
# **Denominación de las Características**
# 
# Así es como se denominan las características:
# `stage`.`parameter_type`.`parameter_name`
# Ejemplo: rougher.input.feed_ag
# 
# Valores posibles para `stage`:
# 
# - `rougher` — flotación
# - `primary_cleaner` — purificación primaria
# - `secondary_cleaner` — purificación secundaria
# - `final` — características finales
# 
# Valores posibles para `parameter_type`:
# 
# - `input` — parámetros de la materia prima
# - `output` — parámetros del producto
# - `state` — parámetros que caracterizan el estado actual de la etapa
# - `calculation` — características de cálculo

# ### Comprobar Recuperación

# Zyfra nos a dado la siguiente fórmula para simular el proceso de recuperación:
# 
# $$Recuperacion = \frac{(C*(F-T)}{F*(C-T)}*100$$
# 
# Donde:
# 
# $C=$ Proporción de oro en el concentrado justo después de la flotación (para saber la recuperación del concentrado rougher) / después de la purificación (para saber la recuperación del concentrado final)
# 
# $F=$ La proporción de oro en la alimentación antes de la flotación (para saber la recuperación del concentrado rougher) / en el concentrado justo después de la flotación (para saber la recuperación del concentrado final)
# 
# $T=$ La proporción de oro en las colas rougher justo después de la flotación (para saber la recuperación del concentrado rougher) / después de la purificación (para saber la recuperación del concentrado final)

# Debemos comprobar si la característica `rougher.output.recovery` es correcta.

# In[9]:


## Recuperación del Concentrado / Recuperación del Concentrado Final
C = (zyfra_train['rougher.output.concentrate_au'].mean()
     / zyfra_train['final.output.concentrate_au'].mean()
    )
## Recuperación en la Alimentación / Recuperación del Concentrado Final
F = (zyfra_train['rougher.input.feed_au'].mean() 
     / zyfra_train['rougher.output.concentrate_au'].mean()
    )
## Recuperación en las Colas / Recuperación Concentrado Final
T = (zyfra_train['primary_cleaner.output.tail_ag'].mean() 
     / zyfra_train['secondary_cleaner.output.tail_au'].mean()
     / zyfra_train['final.output.tail_au'].mean()
    )

## < Aplicamos cálculo de la formula >
recuperacion = (C*(F-T) / F*(C-T))*100
## < Rougher Ouput Recovery en el Set de Entrenamiento>
RoR = zyfra_train['rougher.output.recovery'].mean()

## < Calcular el Error Absoluto Medio>
MAE = recuperacion - RoR


print(f'El valor de C es: {C}')
print(f'El valor de F es: {F}')
print(f'El valor de T es: {T}')
print('----------------------------------------------------------')    
print(f'La recuperación de Oro del set de entrenamiento es: {RoR}')
print(f'La recuperación de Oro calculada es: {recuperacion}')
print(f'El valor del Error Absoluto Medio es: {MAE}')


# ### Características No Disponibles

# In[10]:


## < ¿Qué columnas hay en train que no hay en test..? >
set(zyfra_train.columns) - set(zyfra_test.columns)


# ### Preprocesamiento de Datos
# 
# Datos Nulos en Datasets.
# 
# - Vamos a identificar los datos nulos de manera programática.
# - Desarrollaremos una función para hacerlo.
# - Imputaremos valores con el uso de herramientas de sklearn.

# In[11]:


def detect_null_col(df):
    """
    Función para detectar valores nulos
    """
    null_columns = []
    nulls = df.isna().sum()
    for idx, value in zip(nulls.index, nulls):
        if value > 0:
            null_columns.append(idx)
    return null_columns

null_columns = detect_null_col(zyfra_train)

def impute_columns(df_train, df_test, null_columns, imputation_method = "mean"):
    input_dict = {"mean": df_train[null_columns].mean().to_dict(),
                "median": df_train[null_columns].median().to_dict()}

    return df_train.fillna(value = input_dict[imputation_method]), df_test.fillna(value = input_dict[imputation_method])


# In[12]:


zyfra_train_imp, zyfra_test_imp = impute_columns(zyfra_train, zyfra_test, null_columns, imputation_method="median")

dim = zyfra_train_imp.shape
dim1= zyfra_test_imp.shape

print(f'La dimensión del set de entrenamiento es: {dim}')
print(f'La dimensión del set de prueba es: {dim1}')


# In[13]:


nulo = zyfra_train_imp.isna().sum().sum()
null = zyfra_train_imp.isna().sum().sum()

print(f'La cantidad de datos nulos en el set de entrenamiento es: {nulo}')
print(f'La cantidad de datos nulos en el set de prueba es: {null}')


# **Observación**
# 
# Hemos observado las siguientes anomalías:
# 
# - Tenemos un problema de tamaños en las columnas el set de entrenamiento tiene 87 columnas y el set de prueba tiene solo 53
# - Tuvimos un problema de valores ausentes que ya corregimos.
# - Comprobamos la recuperación pero obtivimos un valor ligeramente distinto.
# 
# En los próximos pasos vamos analizar los datasets de prueba y entrenamiento para corroborar que estén correctos.

# ## || Etapa Dos: Analisis de Datos.
# 
# ---

# **Concentraciones de Au, Ag, Pb**
# 
# Vamos analizar las concentraciones de estos elementos, recordemos que la purificación se divide en dos etapas, ahora vamos a crear una función que nos ayude a simplificar los resultados y a obtener su gráfica.

# In[14]:


## < Función para obtener resultados y gráficar >
def purification_results(data, element, first_step, second_step):
    first = data[first_step]
    print(f'Purificación de {element} Primera Etapa:\n\n{first.describe()}')
    first.hist(label='Primera Etapa', alpha=0.9, figsize=(10,5), color='darkblue')
    print('-----------------------------------------------------------')
    second = data[second_step]
    print(f'Purificación de {element} Segunda Etapa:\n\n{second.describe()}')
    second.hist(label='Segunda Etapa', alpha=0.5, figsize=(10,5), color='orange')
    
    plt.title(f'Purificación de {element}')
    plt.ylabel('Frecuencia')
    plt.xlabel('Cantidad de Concentrado Roughter')
    plt.legend()
    plt.show()
    
    box = data[[first_step, second_step]]
    box.boxplot(figsize=(10,5))
    plt.ylabel('Cantidad de Concentrado Roughter')
    plt.xlabel('Etapas')
    plt.show()


# Purificación del Oro.

# In[15]:


purification_results(zyfra_train_imp,'Oro', 'primary_cleaner.output.concentrate_au', 'secondary_cleaner.output.tail_au')


# Purificación de Plata

# In[16]:


purification_results(zyfra_train_imp,'Plata','primary_cleaner.output.concentrate_ag', 'secondary_cleaner.output.tail_ag')


# Purificación de Plomo

# In[17]:


purification_results(zyfra_train_imp,'Plomo', 'primary_cleaner.output.concentrate_pb', 'secondary_cleaner.output.tail_pb')


# Al momento de procesar nuestros resultados observamos lo siguiente:
# 
# - Los tres elementos tienen valores atípicos (outliers)
# - Con los histogramas se puede observar que hay valores cercanos a cero.
# - Los boxplot nos indican cuáles son estos valores.
# 
# **Hallazgos**
# 
# Nos hemos dado cuenta de que estos valores atípicos cercanos a cero se encuentran en la primera etapa de purificación, esto no debería ocurrir puesto que se trata solo de la primera etapa de limpieza. Estos valores pueden afectar a nuestro modelo por lo que vamos a filtrarlos.
# 
# Para el caso de la plata y el plomo no será necesario ya que nuestro modelo no necesita predecir o trabajar con estos estos valores.

# In[18]:


zyfra_train_filter = zyfra_train_imp[(zyfra_train_imp['primary_cleaner.output.concentrate_au'] >= 24) & (zyfra_train_imp['primary_cleaner.output.concentrate_au'] <= 42) & (zyfra_train_imp['secondary_cleaner.output.tail_au'] <= 7)]


# Nuevamente aplicamos el análisis.

# In[19]:


purification_results(zyfra_train_filter, 'Oro', 'primary_cleaner.output.concentrate_au', 'secondary_cleaner.output.tail_au')


# De esta manera nuestro modelo podrá predecir mucho mejor los resultados que esperamos obtener.

# ### Comparación de concentraciones en la Alimentación.
# 
# Vamos a comparar las distribuciones del tamaño de las partículas de la alimentación en el conjunto de entrenamiento y en el conjunto de prueba. Si las distribuciones varían significativamente, la evaluación del modelo no será correcta.

# In[20]:


def feed_results(data_train, data_test, column):
    train = data_train[column]
    print(f'Concentraciones de Alimentación en Entrenamiento:\n\n{train.describe()}')
    train.hist(label='Alimentación en Entrenamiento', alpha=0.8, figsize=(10,5), color='orange')
    print('-----------------------------------------------------------')
    test = data_test[column]
    print(f'Concentraciones de Alimentación en Prueba:\n\n{test.describe()}')
    test.hist(label='Alimentación en Prueba', alpha=0.6, figsize=(10,5), color='black')
    plt.title('Concentraciones en la Alimentación')
    plt.ylabel('Frecuencia')
    plt.xlabel('Cantidad')
    plt.legend()
    plt.show()


# In[21]:


feed_results(zyfra_train_imp, zyfra_test_imp, 'rougher.input.feed_size')


# Después de evaluar resultados podemos ver que las distribuciones son practicamente iguales para ambos sets por lo tanto podemos inferir que nuestro modelo funciona correctamente.

# ### Concentraciones Totales
# 
# Considerando las concentraciones totales de todas las sustancias en las diferentes etapas: materia prima, concentrado rougher y concentrado final. Vamos a responder las siguientes incognitas: ¿Se observa algún valor anormal en la distribución total? Si es así, ¿merece la pena eliminar esos valores de ambas muestras?
# 
# Para responder la pregunta primero vamos a desarrollar una función que nos ayude a ver el espectro completo.

# In[22]:


def total_concentrate(data, element):
    
    feed_in   = zyfra_train_imp[f'rougher.input.feed_{element}'].describe()
    purific_1 = zyfra_train_imp[f'primary_cleaner.output.concentrate_{element}'].describe()
    purific_2 = zyfra_train_imp[f'secondary_cleaner.output.tail_{element}'].describe()
    final     = zyfra_train_imp[f'final.output.concentrate_{element}'].describe()
    
    print(f'Concentración en la Alimentación de {element} \n\n{feed_in}')
    print('------------------------------------------------------------')
    print(f'Concentración en la Purificación 1 de {element} \n\n{purific_1}')
    print('------------------------------------------------------------')
    print(f'Concentración en la Purificación 2 de {element} \n\n{purific_2}')
    print('------------------------------------------------------------')
    print(f'Concentración en la Purificación final de {element} \n\n{final}')
    
    columns= [f'rougher.input.feed_{element}',
             f'primary_cleaner.output.concentrate_{element}',
             f'secondary_cleaner.output.tail_{element}',
             f'final.output.concentrate_{element}']
    
    for col in columns:
        data[col].plot(kind='hist', alpha=0.5, bins=100, figsize=(10,5))
    plt.legend(columns);


# Concentraciones Totales de Oro

# In[23]:


total_concentrate(zyfra_train_imp, 'au')


# Concentraciones Totales de Plata

# In[24]:


total_concentrate(zyfra_train_imp, 'ag')


# Concentraciones Totales de Plomo

# In[25]:


total_concentrate(zyfra_train_imp, 'pb')


# **Observaciones**
# 
# Después de analizar la información podemos ver que existen una gran cantidad de elementos que tienen cero residuos, al inicio pensamos que se traban de valores atípicos pero leyendo la descripción de nuestro proyecto podemos inferir que se trata de elementos como el oro con menor cantidad de residuos no valiosos, es decir, las distribuciones que vemos en las gráficas son la cantidad de residuos de cada elemento y mientras más se acercan a cero el mineral es más valioso.
# 
# Esto explica el por qué tenemos alrededor de 1000 a 2000 valores de cada material (oro, plata, plomo) cerca del valor de cero.

# ## || Construir un Modelo
# ---
# 
# En este punto vamos a desarrollar dos modelos para averiguar cuál sería la mejor opción para la tarea de predicción que Zyfra necesita.
# 
# Para comenzar vamos a desarrollar una función que servirá para evaluar la calidad de nuestro modelo. Este parámetro se denomina sMAPE y es una función que no encontramos en las librerías de sklearn por lo tanto debemos desarrollarla de forma manual.
# 
# Esta es la fórmula:
# 
# $$sMAPE = \frac{1}{N}\sum_{i=1}^{n}\frac{|yi - \widetilde{y}i|}{(|yi|+|\widetilde{y}i|)/2}$$

# In[26]:


## < Funcion para calcular SMAPE >
def smape(y_target , y_prediction):
    num = (y_target - y_prediction).abs()
    den = (y_target + y_prediction).abs()/2
    return (num/den).mean()

def final_smape(y_target, y_prediction):
    rougher  = smape(y_target[0], y_prediction[1])
    recovery = smape(y_target[1], y_prediction[1])
    return 0.25*rougher + 0.75*recovery


# ### Data Leakage

# Antes de proceder a realizar nuestro modelo debemos resolver el tema de los sets de zyfra ya que ambos (prueba y entrenamiento) contienen columnas diferentes. Lo que sabemos es que para desarrollar un nuevo modelo las columnas de ambos sets deben ser del mismo tamaño.
# 
# Hemos observado que el set de prueba no contiene algunas columnas que el set de entrenamiento si tiene. Esto se debe a que probablemente tenemos una fuga de información, es decir, con el set de prueba zyfra intenta que no exista información que no debería estar ahí puesto que se trata de información del futuro.
# 
# Con información del futuro nos referimos a variables que son las que se deben predecir pero si estas ya tienen un registro entonces no hay algún valor qué predecir puesto que ya existe. Por lo tanto debemos eliminar o quitar estos registros como el caso del set de prueba.

# In[27]:


## <Asignación de Características y Objetivos>

features = zyfra_test_imp.columns
target = ["rougher.output.recovery", "final.output.recovery"]

## <Train Test Split>

# Entrenamiento...
X_train = zyfra_train_imp[features].reset_index(drop=True)
y_train = zyfra_train_imp[target].reset_index(drop=True)
y_train.columns = [0,1]

# Test...
X_test = zyfra_test_imp[features].reset_index(drop=True)
y_test = zyfra_full.loc[zyfra_test_imp.index, target].reset_index(drop=True)
y_test.columns = [0,1]

X_train.shape, y_train.shape, X_test.shape, y_test.shape


# De esta forma es como hacemos split y obtenemos nuestro set de entrenamiento y nuestro set de prueba a partir de la recuperación de características en el set de datos completo.
# 
# Ahora veamos si tiene datos nulos.

# In[28]:


X = X_train.fillna(method = "ffill")
X_test = X_test.fillna(method = "ffill")
y = y_train.fillna(method = "ffill")
y_test = y_test.fillna(method = "ffill")

X.isna().sum().sum(), X_test.isna().sum().sum(), y.isna().sum().sum(), y_test.isna().sum().sum()


# ### Validación Cruzada / Cross-validation

# Ahora vamos a entrenar un nuevo modelo con las características igualadas para obtener un resultado más preciso. Este modelo debe ser de regresión dadas las características que tiene nuestro set de datos. Una clasificación no sería posible implementarla.

# In[29]:


## Modelo 1: Random Forest Regressor

rfr = RandomForestRegressor(random_state=42, n_jobs=-1, max_depth=7)
kf  = KFold(n_splits=5) # instanciarlo, generador... utilizando un loop

start_time = time.time()
score = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start = 1): 
    X_train = X.loc[train_idx].reset_index(drop = True)
    y_train = y.loc[train_idx].reset_index(drop = True) 
    X_val = X.loc[val_idx].reset_index(drop = True)
    y_val = y.loc[val_idx].reset_index(drop = True)
    
    rfr.fit(X_train, y_train)
    y_prediction = pd.DataFrame(rfr.predict(X_val))
    print(f"Score for fold {fold}: {final_smape(y_val, y_prediction)}")
    score.append(final_smape(y_val, y_prediction))
    
SMAPE = np.mean(score)
print(f'sMAPE Final:{SMAPE}')
print("Este modelo se ajustó en:", (time.time() - start_time), "segundos")


# In[30]:


## Modelo 2: Linear Regression

lr = LinearRegression(n_jobs=-1)
kf  = KFold(n_splits=5) # instanciarlo, generador... utilizando un loop

start_time = time.time()
score = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start = 1): 
    X_train = X.loc[train_idx].reset_index(drop = True)
    y_train = y.loc[train_idx].reset_index(drop = True) 
    X_val = X.loc[val_idx].reset_index(drop = True)
    y_val = y.loc[val_idx].reset_index(drop = True)
    
    rfr.fit(X_train, y_train)
    y_prediction = pd.DataFrame(rfr.predict(X_val))
    print(f"Score for fold {fold}: {final_smape(y_val, y_prediction)}")
    score.append(final_smape(y_val, y_prediction))
    
SMAPE = np.mean(score)
print(f'sMAPE Final:{SMAPE}')
print("Este modelo se ajustó en:", (time.time() - start_time), "segundos")


# In[31]:


## Modelo 3: Decision Tree Regresor

dtr = DecisionTreeRegressor(max_depth=5)
kf  = KFold(n_splits=5) # instanciarlo, generador... utilizando un loop

start_time = time.time()
score = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start = 1): 
    X_train = X.loc[train_idx].reset_index(drop = True)
    y_train = y.loc[train_idx].reset_index(drop = True) 
    X_val = X.loc[val_idx].reset_index(drop = True)
    y_val = y.loc[val_idx].reset_index(drop = True)
    
    dtr.fit(X_train, y_train)
    y_prediction = pd.DataFrame(dtr.predict(X_val))
    print(f"Score for fold {fold}: {final_smape(y_val, y_prediction)}")
    score.append(final_smape(y_val, y_prediction))
    
SMAPE = np.mean(score)
print(f'sMAPE Final:{SMAPE}')
print("Este modelo se ajustó en:", (time.time() - start_time), "segundos")


# ## Conclusiones
# 
# ---
# 
# Durante este proyecto implementamos nuevas habilidades que no habíamos descubierto antes, calculamos el erro absoluto medio (EAM) de un modelo que ya se encontraba en su fase de split que fue proporcionado por la empresa zyfra.
# 
# Se evaluaron y se trataron diversas anomalías como los valores nulos y las columnas faltantes, además los valores nulos los rellenamos con la media y usamos también una nueva técnica que es parte del método de fillna llamado "ffill" que toma los valores del índice, que en este caso es una serie de tiempo, y sustituye los valores ausentes con los inmediatos superiores a su indice.
# 
# Además revisamos las distribuciones de los datos proporcionados en los sets y realizamos algunos ajustes para darnos cuenta del verdadero valor real en las gráficas.
# 
# Finalmente construimos un modelo de predicción a partir de los sets proporcionados por zyfra donde encontramos columnas incompletas en el set de prueba mismas que fueron extraídas del set completo para obtener la misma cantidad que el set de entrenamiento.
# 
# También solucionamos algunos inconvenientes con el Data Leakage para evitar la fuga de información y mejorar nuestros resultados.
# 
# Por último evaluamos algunos modelos con la nueva métrica sMAPE donde obtuvimos en lo general un error de alrededor del 15% lo cual indica que el modelo no es malo. Quizás se pueda mejorar.
# 
# Esperemos que esta investigación sea de ayuda para Zyfra en su tarea de extracción de minerales principalmente el oro y que este modelo sirva para ayudarlos con sus labores.
