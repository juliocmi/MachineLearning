#!/usr/bin/env python
# coding: utf-8

# # Álgebra Líneal, Sure Tomorrow
# ---
# 
# **NoteBook By:** 
# 
# **Data Scientist:** Julio César Martínez Izaguirre

# # Tabla de Contenido
# ---
# 
# 1. Inicialización.
# 2. Carga de Datos.
# 3. Análisis Exploratorio de Datos.
# 4. Prueba de la Ofuscación de Datos.
# 5. Prueba de la Regresión Líneal.
# 6. Apéndice A: Escribir Fórmulas en Notebook.
# 7. Apéndice B: Propiedades de las Matrices.
# 8. Bibliografía.
# 9. Agradecimientos.

# # Introducción
# ---
# 
# **Datos Curiosos Sobre los Seguros**
# 
# En la antigua Babilonia se indemnizaba a las esposas de los guerreros en caso de muerte. Quizá este sea uno de los primeros ejemplos de un seguro de vida como prestación laboral.
# 
# La aseguradora londinense Lloyd´s firmó con una empresa cinematográfica una póliza de vida que cubría a todos los espectadores de sus películas en caso de que alguno sufriera una muerte repentina por un ataque de risa. 
# 
# Suena a película de ciencia ficción, pero hay aseguradoras que ofrecen seguros de vida en caso de apocalipsis zombi o incluso si hay una invasión extraterrestre; de hecho, algunas compañías han llegado a ofrecer coberturas en caso de maldiciones, hechizos o fenómenos paranormales.
# 
# La primera forma oficial de seguros fueron los fenicios quienes crearon las primeras empresas aseguradoras relacionadas con el comercio marítimo. Si un barco mercante no llegaba a puerto, su valor económico era devuelto en función de la mercancía transportada a cambio de que se realizará un pago antes de que el barco partiera.
# 
# La primera póliza de la que se tiene constancia está fechada en el año 1347. Se redactó y firmó en Génova para asegurar contra posibles accidentes, naufragios o asaltos piratas al navío mercante Santa Bárbara que cubría el trayecto desde Génova a Mallorca. 
# 
# En México en 1789 se constituyó la primera empresa de seguros, llamada “Compañía de Seguros Marítimos de Nueva España”. Esto para satisfacer la protección marítima en el Puerto de Veracruz, que a fines del siglo XVIII gozaba de gran prosperidad comercial.
# 
# Ésta daría paso al establecimiento de otras aseguradoras y, en general, al desarrollo de la actividad aseguradora en nuestro país(México).
# 
# **ACTUALIDAD Y RIESGOS**
# 
# En el tercer trimestre de 2022 se registró un producto interno bruto de $1.07B MX, evidenciando un alza de 1.61% con respecto al trimestre anterior.
# 
# Según DENUE 2022, Compañías de Seguros y Fianzas registró 1,994 unidades económicas. Las entidades federativas con mayor número de unidades económicas fueron Ciudad de México (241), Jalisco (137) y Nuevo León (119).
# 
# En cuanto a riesgos México ocupó el primer lugar de la región con 67% de intentos de ataque en 2021.
# 
# Por estas razones las compañías de seguros están comprometidas con sus clientes ofreciéndoles servicios de alta calidad y para asegurarse de que así suceda ahora las compañías hacen uso de la tecnología para mejorar los resultados que obtienen frente a sus clientes. Este es el caso de nuestro proyecto en el cuál trabajaremos a continuación.

# # Descripción del Proyecto
# 
# ---

# La compañía de seguros **Sure Tomorrow** quiere resolver varias tareas con la ayuda de machine learning y nos pide que evaluemos esa posibilidad.
# - Tarea 1: encontrar clientes que sean similares a un cliente determinado. Esto ayudará a los agentes de la compañía con el marketing.
# - Tarea 2: predecir la probabilidad de que un nuevo cliente reciba una prestación del seguro. ¿Puede un modelo de predictivo funcionar mejor que un modelo dummy?
# - Tarea 3: predecir el número de prestaciones de seguro que un nuevo cliente pueda recibir utilizando un modelo de regresión lineal.
# - Tarea 4: proteger los datos personales de los clientes sin afectar al modelo del ejercicio anterior. Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento u ofuscación de datos. Pero los datos deben protegerse de tal manera que no se vea afectada la calidad de los modelos de machine learning. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.
# 
# Es necesario desarrollar un algoritmo de transformación de datos que dificulte la recuperación de la información personal si los datos caen en manos equivocadas. Esto se denomina enmascaramiento de datos u ofuscación de datos. Pero los datos deben protegerse de tal manera que la calidad de los modelos de machine learning no se vea afectada. No es necesario elegir el mejor modelo, basta con demostrar que el algoritmo funciona correctamente.

# # Preprocesamiento y exploración de datos
# 
# ## Inicialización

# In[1]:


#pip install scikit-learn --upgrade


# In[2]:


# importando librerías
import numpy as np
import pandas as pd
import math
import scipy

import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.linear_model
import sklearn.metrics 
import sklearn.neighbors
import sklearn.preprocessing

from sklearn.model_selection import train_test_split

from IPython.display import display


# ## Carga de datos

# Cargar datos y un primer vistaso a los errores más obvios.

# In[3]:


df = pd.read_csv('/datasets/insurance_us.csv')
df.head()


# Renombramos las columnas para que el código se vea más coherente con su estilo.

# In[4]:


df = df.rename(columns={'Gender': 'gender', 'Age': 'age', 'Salary': 'income', 'Family members': 'family_members', 'Insurance benefits': 'insurance_benefits'})
df.head()


# revisión de columnas y valores

# In[5]:


df.info()


# In[6]:


# la columna 'age' es de tipo float pero debería ser int por su naturaleza, vamos a cambiarlo
df['age'] = df['age'].astype(int)
df.info()


# Ahora veamos las estadísticas descriptivas de los datos

# In[7]:


df.describe()


# **Observaciones**
# 
# * Todo parece estar bien con los datos, no hay valores ausentes o nulos. 
# 
# * El género tiene un promedio del 50% es decir la mitad de nuestro dataset son hombres y la mitad mujeres. 
# 
# * El ingreso promedio es de 39000. 
# 
# * La taza de integrantes de una familia va de 1 a 6 integrantes siendo 1 el promedio.
# 
# * Los beneficios recibidos practicamente son nulos.

# ## Análisis exploratorio de datos

# Vamos a comprobar rápidamente si existen determinados grupos de clientes observando el gráfico de pares.

# In[8]:


g = sns.pairplot(df, kind='hist')
g.fig.set_size_inches(12, 12)


# De acuerdo, es un poco complicado detectar grupos obvios (clústeres) ya que es difícil combinar diversas variables simultáneamente (para analizar distribuciones multivariadas). Ahí es donde LA y ML pueden ser bastante útiles.

# # Tarea 1. Clientes similares

# En el lenguaje de ML, es necesario desarrollar un procedimiento que devuelva los k vecinos más cercanos (objetos) para un objeto dado basándose en la distancia entre los objetos.
# 
# Los temas clave para resolver esta tarea residen en:
# - Distancia entre vectores -> Distancia Manhattan
# 
# Para resolver la tarea, podemos probar diferentes métricas de distancia.

# Vamos a escribir una función que devuelva los k vecinos más cercanos para un $n^{th}$ objeto basándose en una métrica de distancia especificada. A la hora de realizar esta tarea no debe tenerse en cuenta el número de prestaciones de seguro recibidas.
# Podemos utilizar una implementación ya existente del algoritmo kNN de scikit-learn (consulta [el enlace](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)) o la propia implementación.
# 
# Probaremos esta función para cuatro combinaciones de dos casos:
# 
# - Escalado
#   - los datos no están escalados
#   - los datos se escalan con el escalador [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
# - Métricas de distancia
#   - Euclidiana
#   - Manhattan
# 
# **Responderemos a estas preguntas:**
# - ¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?
# - ¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?

# In[9]:


feature_names = ['gender', 'age', 'income', 'family_members']


# In[10]:


def get_knn(df, n, k, metric):
    
    """
    Devuelve los k vecinos más cercanos

    :param df: DataFrame de pandas utilizado para encontrar objetos similares dentro del mismo lugar    
    :param n : número de objetos para los que se buscan los vecinos más cercanos    
    :param k : número de vecinos más cercanos a devolver
    :param métrica: nombre de la métrica de distancia    """

    nbrs = sklearn.neighbors.NearestNeighbors(metric=metric).fit(df[feature_names])  
    nbrs_distances, nbrs_indices = nbrs.kneighbors([df.iloc[n][feature_names]], k, return_distance=True)
    
    df_res = pd.concat(
        [df.iloc[nbrs_indices[0]], 
        pd.DataFrame(
            nbrs_distances.T,
            index=nbrs_indices[0],
            columns=['distance']
        )], axis=1
    )
    
    return df_res


# Escalar datos.

# In[11]:


feature_names = ['gender', 'age', 'income', 'family_members']
transformer_mas = sklearn.preprocessing.MaxAbsScaler().fit(df[feature_names].to_numpy())

df_scaled = df.copy()
df_scaled.loc[:, feature_names] = transformer_mas.transform(df[feature_names].to_numpy())


# In[12]:


df_scaled.sample(5)


# Ahora, vamos a obtener registros similares para uno determinado, para cada combinación

# Combinación 1

# In[13]:


## Combinación 1: Índice 75, distancia euclidiana por edad sin escalar
comb_1 = get_knn(df, 75, 10, 'euclidean')
comb_1


# In[14]:


sns.scatterplot(data=comb_1, x="age", y="distance")
sns.set_theme(style='darkgrid');


# Combinación 2

# In[15]:


## Combinación 2: Indice 75, distancia manhattan por edad sin escalar
comb_2 = get_knn(df, 75, 10, 'manhattan')
comb_2


# In[16]:


sns.scatterplot(data=comb_2, x="age", y="distance")
sns.set_theme(style='darkgrid');


# Combinación 3

# In[17]:


## Combinación 3: Índice 75, distancia euclidiana por edad, datos escalados.
comb_3 = get_knn(df_scaled, 75, 10, 'euclidean')
comb_3


# In[18]:


sns.scatterplot(data=comb_3, x="age", y="distance")
sns.set_theme(style='darkgrid');


# Combinación 4

# In[19]:


## Combinación 4: Índice 75, distancia manhattan por edad, datos escalados.
comb_4 = get_knn(df_scaled, 75, 10, 'manhattan')
comb_4


# In[20]:


sns.scatterplot(data=comb_4, x="age", y="distance")
sns.set_theme(style='darkgrid');


# Respuestas a las preguntas

# **¿El hecho de que los datos no estén escalados afecta al algoritmo kNN? Si es así, ¿cómo se manifiesta?** 
# 
# Los datos que no estan escalados afectan al agoritmo KNN de tal forma que se ven afectadas las características por el peso que el algoritmo les da. Por ejemplo las edades van con números de los 18 a los 65 años mientras que el ingreso va de los 5 mil a los 79 mil, son números muy diferentes.
# 
# Si eso lo transformamos a una escala de unos y ceros es más fácil para el algoritmo ajustar todas las clases.

# **¿Qué tan similares son los resultados al utilizar la métrica de distancia Manhattan (independientemente del escalado)?** 
# 
# Los resultados entre la distancia eclidiana y manhattan suelen ser ligeramente distintos para este ejemplo.

# # Tarea 2. ¿Es probable que el cliente reciba una prestación del seguro?

# En términos de machine learning podemos considerarlo como una tarea de clasificación binaria.

# Con el valor de `insurance_benefits` superior a cero como objetivo, evaluaremos si el enfoque de clasificación kNN puede funcionar mejor que el modelo dummy.
# Instrucciones:
# - Construiremos un clasificador basado en KNN y mediremos su calidad con la métrica F1 para k=1...10 tanto para los datos originales como para los escalados. Sería interesante observar cómo k puede influir en la métrica de evaluación y si el escalado de los datos provoca alguna diferencia. 
# 
# - Construiremos un modelo dummy que, en este caso, es simplemente un modelo aleatorio. Debería devolver "1" con cierta probabilidad. Probemos el modelo con cuatro valores de probabilidad: 0, la probabilidad de pagar cualquier prestación del seguro, 0.5, 1.
# La probabilidad de pagar cualquier prestación del seguro puede definirse como
# $$
# P\{\text{prestación de seguro recibida}\}=\frac{\text{número de clientes que han recibido alguna prestación de seguro}}{\text{número total de clientes}}.
# $$
# 
# Dividiremos todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30.

# In[21]:


# сalculando el objetivo
df['insurance_benefits_received'] = df['insurance_benefits'] > 0


# In[22]:


# comprobamos el desequilibrio de clases con value_counts()
df['insurance_benefits_received'].value_counts()


# In[23]:


def eval_classifier(y_true, y_pred):
    
    f1_score = sklearn.metrics.f1_score(y_true, y_pred)
    print(f'F1: {f1_score:.2f}')
    
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='all')
    print('Matriz de confusión')
    print(cm)


# In[24]:


# generar la salida de un modelo aleatorio

def rnd_model_predict(P, size, seed=42):

    rng = np.random.default_rng(seed=seed)
    return rng.binomial(n=1, p=P, size=size)


# In[25]:


for P in [0, df['insurance_benefits_received'].sum() / len(df), 0.5, 1]:

    print(f'La probabilidad: {P:.2f}')
    y_pred_rnd = rnd_model_predict(P, len(df))
        
    eval_classifier(df['insurance_benefits_received'], y_pred_rnd)
    
    print()


# **Creación de Modelo de Clasificación KNN**
# 
# Modelo de clasificación para datos no escalados.

# In[26]:


random = np.random.RandomState(42)


# In[27]:


def knn_classifier(data):    
    k_range = range(1,11)

    ## Seleccionamos Objetivos
    X = data.drop(['insurance_benefits_received', 'insurance_benefits'], axis=1)
    y = data['insurance_benefits_received']

    ## Hacemos Split de Datos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.3, 
        random_state=random
    )
    
    ## Bucle para los 10 vecinos más cercanos
    for k in k_range:
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train, y_train)
        y_pred   = knn.predict(X_test)
        f1_score = sklearn.metrics.f1_score(y_test, y_pred)
        print("k_neighbors:", k)
        eval_classifier(y_test, y_pred)
        plt.scatter(
            x=k,
            y=f1_score
        )
        
    ## Graficamos
    plt.title('Metricas n_neighbors 1 - 10')
    plt.xlabel('K Neighbors')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.xticks([0,2,4,6,8,10])
    plt.style.use('ggplot')
    plt.show();


# In[28]:


knn_classifier(df)


# Modelo de clasificación para datos escalados.

# In[29]:


df_scaled['insurance_benefits_received'] = df_scaled['insurance_benefits'] > 0


# In[30]:


knn_classifier(df_scaled)


# Podemos comprobar que al realizar un escalamiento de los datos obtenemos una métrica bastante buena para nuestro equilibrio armónico de recall y precision.

# # Tarea 3. Regresión (con regresión lineal)

# Con `insurance_benefits` como objetivo, evalúar cuál sería la RECM de un modelo de regresión lineal.

# Construye tu propia implementación de regresión lineal. Comprobaremos la RECM tanto para los datos originales como para los escalados. ¿Podremos ver alguna diferencia en la RECM con respecto a estos dos casos?
# 
# Denotemos- $X$: matriz de características; cada fila es un caso, cada columna es una característica, la primera columna está formada por unidades
# - $y$ — objetivo (un vector)
# - $\hat{y}$ — objetivo estimado (un vector)
# - $w$ — vector de pesos
# 
# La tarea de regresión lineal en el lenguaje de las matrices puede formularse así:
# $$
# y = Xw
# $$
# 
# El objetivo de entrenamiento es entonces encontrar esa $w$ w que minimice la distancia L2 (ECM) entre $Xw$ y $y$:
# 
# $$
# \min_w d_2(Xw, y) \quad \text{or} \quad \min_w \text{MSE}(Xw, y)
# $$
# 
# Parece que hay una solución analítica para lo anteriormente expuesto:
# $$
# w = (X^T X)^{-1} X^T y
# $$
# 
# La fórmula anterior puede servir para encontrar los pesos $w$ y estos últimos pueden utilizarse para calcular los valores predichos
# $$
# \hat{y} = X_{val}w
# $$

# Divide todos los datos correspondientes a las etapas de entrenamiento/prueba respetando la proporción 70:30. Utiliza la métrica RECM para evaluar el modelo.

# In[31]:


class MyLinearRegression:
    
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        self.weights = np.linalg.inv(X2.T.dot(X2)).dot(X2.T).dot(y)
        self.w0 = 0

    def predict(self, X):
        # añadir las unidades
        X2 = np.append(np.ones([len(X), 1]), X, axis=1)
        y_pred = X2.dot(self.weights) + self.w0
        
        return y_pred


# In[32]:


def eval_regressor(y_true, y_pred):
    
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse:.2f}')
    
    r2_score = math.sqrt(sklearn.metrics.r2_score(y_true, y_pred))
    print(f'R2: {r2_score:.2f}')    


# Implementación de la clase con datos sin escalar

# In[33]:


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(pd.Series(lr.weights))
print()

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# Comprobamos que nuestra clase funciona correctamente usando la librería sklearn

# In[34]:


## Instanciar Modelo y Entrenar
lrm = sklearn.linear_model.LinearRegression().fit(X_train, y_train)

## Obtener Predicciones
y_pred = lrm.predict(X_test)

## Métricas de Evaluación
eval_regressor(y_test, y_pred)


# Implementación con datos escalados

# In[35]:


Xs = df_scaled[['age', 'gender', 'income', 'family_members']].to_numpy()
ys = df_scaled['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(pd.Series(lr.weights))
print()

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# No se perciben cambios en las métricas de evaluación RMSE y R2 respecto a los datos escalados y no escalados.

# # Tarea 4. Ofuscar datos

# Lo mejor es ofuscar los datos multiplicando las características numéricas (recuerda que se pueden ver como la matriz $X$) por una matriz invertible $P$. 
# 
# $$
# X' = X \times P
# $$
# 
# Trata de hacerlo y comprueba cómo quedarán los valores de las características después de la transformación. Por cierto, la propiedad de invertibilidad es importante aquí, así que asegúrate de que $P$ sea realmente invertible.
# 
# Puedes revisar la lección 'Matrices y operaciones matriciales -> Multiplicación de matrices' para recordar la regla de multiplicación de matrices y su implementación con NumPy.

# In[36]:


personal_info_column_list = ['gender', 'age', 'income', 'family_members']
df_pn = df[personal_info_column_list]


# In[37]:


X = df_pn.to_numpy()


# Generar una matriz aleatoria $P$.

# In[38]:


rng = np.random.default_rng(seed=42)
P = rng.random(size=(X.shape[1], X.shape[1]))


# Comprobar que la matriz P sea invertible

# In[39]:


## Primera Matriz
X, X.shape


# In[40]:


## Segunda Matriz
P, P.shape


# In[41]:


## Comprobar que P es invertible.

I = scipy.linalg.inv(P)
I, I.shape


# In[42]:


## Multiplicamos
X1 = X @ I
X1, X1.shape


# ¿Puedes adivinar la edad o los ingresos de los clientes después de la transformación?

# *No. Es complicado visualizar los datos requeridos a simple vista.*

# ¿Puedes recuperar los datos originales de $X'$ si conoces $P$? Intenta comprobarlo a través de los cálculos moviendo $P$ del lado derecho de la fórmula anterior al izquierdo. En este caso las reglas de la multiplicación matricial son realmente útiles

# In[43]:


## Transponemos los datos, sin la transposición no es posible realizar la multiplicación.
p = (I @ X.T).T
p, p.shape


# Muestra los tres casos para algunos clientes
# - Datos originales
# - El que está transformado
# - El que está invertido (recuperado)

# In[44]:


## Datos Originales.
pd.DataFrame(X, columns=personal_info_column_list)


# In[45]:


## Datos invertidos
pd.DataFrame(X1, columns=personal_info_column_list)


# In[46]:


## Datos Recuperados
pd.DataFrame(p, columns=personal_info_column_list).head()


# Seguramente puedes ver que algunos valores no son exactamente iguales a los de los datos originales. ¿Cuál podría ser la razón de ello?

# *Se debe a que la multiplicación de matrices entrega resultados diferentes*

# ## Prueba de que la ofuscación de datos puede funcionar con regresión lineal

# En este proyecto la tarea de regresión se ha resuelto con la regresión lineal. Tu siguiente tarea es demostrar _analíticamente_ que el método de ofuscación no afectará a la regresión lineal en términos de valores predichos, es decir, que sus valores seguirán siendo los mismos. ¿Lo puedes creer? Pues no hace falta que lo creas, ¡tienes que que demostrarlo!

# Entonces, los datos están ofuscados y ahora tenemos $X \times P$ en lugar de tener solo $X$. En consecuencia, hay otros pesos $w_P$ como
# $$
# w = (X^T X)^{-1} X^T y \quad \Rightarrow \quad w_P = [(XP)^T XP]^{-1} (XP)^T y
# $$
# 
# ¿Cómo se relacionarían $w$ y $w_P$ si simplificáramos la fórmula de $w_P$ anterior? 
# 
# ¿Cuáles serían los valores predichos con $w_P$? 
# 
# ¿Qué significa esto para la calidad de la regresión lineal si esta se mide mediante la RECM?
# Revisa el Apéndice B Propiedades de las matrices al final del cuaderno. ¡Allí encontrarás fórmulas muy útiles!
# 
# No es necesario escribir código en esta sección, basta con una explicación analítica.

# **Respuesta**

# *La forma en que se relacionan $w$ y $wp$ es que ambas representan el valor de peso de la regresión en ambas fórmulas. Los valores predichos serían todas las observaciones (y) que se multiplican por las matrices previas a Y en la fórmula. Por lo tanto la calidad de la RECM no se ve afectada si comparamos ambas fórmulas.*

# **Prueba analítica**

# Vamos a demostrar lo siguiente

# $$w_P = [(XP)^ T XP] ^ {-1} (XP)^T y$$

# Considerando que $(AB)^{T} = B^{T} A^{T}$ entonces

# $$w_P = [(P)^{T} (X^{T}X) P]^{-1} P^T X^T y$$

# Luego, si... $(AB)^{-1} ) = B^{-1}A^{-1}$

# $$w_P = P^{-1} (X^T X)^{-1} (P^T)^{-1} P^T X^T y$$

# De acuerdo con la propiedad de Identidad: $A^{-1} A = AA^{-1} = I$

# $$w_P = P^{-1} (X^T X)^{-1} I X^T y$$

# A su vez si $IA = AI = A$ entonces...

# $$w_P = P^{-1} (X^T X)^{-1} X^T y$$

# Si tomamos en cuenta que

# $$w = (X^T X)^{-1} X^T y$$

# Podemos concluir que...

# $$w_P = P^{-1} w$$

# ## Prueba de regresión lineal con ofuscación de datos

# Ahora, probemos que la regresión lineal pueda funcionar, en términos computacionales, con la transformación de ofuscación elegida.
# Construye un procedimiento o una clase que ejecute la regresión lineal opcionalmente con la ofuscación. Puedes usar una implementación de regresión lineal de scikit-learn o tu propia implementación.
# Ejecuta la regresión lineal para los datos originales y los ofuscados, compara los valores predichos y los valores de las métricas RMSE y $R^2$. ¿Hay alguna diferencia?

# **Procedimiento**
# 
# - Crea una matriz cuadrada $P$ de números aleatorios.
# - Comprueba que sea invertible. Si no lo es, repite el primer paso hasta obtener una matriz invertible.
# - Utiliza $XP$ como la nueva matriz de características

# Creamos matriz aleatoria

# In[47]:


rdm = np.random.default_rng(seed=42)
pr  = rng.random(size=(df.shape[1], df.shape[1]))
pr


# Comprobamos que sea invertible

# In[48]:


Ipr = np.linalg.inv(pr)
Ipr


# Creamos matriz X*P

# In[49]:


xp = pr.dot(Ipr)
xp


# Creamos Data con matriz aleatoria

# In[50]:


df_cl = pd.DataFrame(xp, columns=df.columns)
df_cl


# Regresión con datos originales

# In[51]:


X = df[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(pd.Series(lr.weights))
print()

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# Regresión con Datos Ofuscados

# In[52]:


X = df_cl[['age', 'gender', 'income', 'family_members']].to_numpy()
y = df_cl['insurance_benefits'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.3, random_state=12345)

lr = MyLinearRegression()

lr.fit(X_train, y_train)
print(pd.Series(lr.weights))
print()

y_test_pred = lr.predict(X_test)
eval_regressor(y_test, y_test_pred)


# # Conclusiones

# Después de pasar un tiempo analizando este proyecto llegamos a lo siguiente:
# 
# - Encontramos clientes que eran similares a un cliente determinado para el dpto de marketing.
# - Predecimos la probabilidad de que un nuevo cliente reciba un beneficio de seguro comparando un modelo ficticio con uno real.
# - Predecimos la cantidad de beneficios de seguro que probablemente recibirá un nuevo cliente utilizando un modelo de regresión lineal.
# - Protegimos los datos personales de los clientes sin romper el modelo de la tarea anterior.
# 
# Desarrollamos un algoritmo de transformación de datos que dificultó la recuperación de la información personal si los datos caen en manos equivocadas. Esto lo denominamos enmascaramiento de datos y se protegieron de tal manera que la calidad de los modelos de machine learning no se vea afectada.

# # Lista de control

# Escribe 'x' para verificar. Luego presiona Shift+Enter.

# - [x]  Jupyter Notebook está abierto
# - [x]  El código no tiene errores 
# - [x]  Las celdas están ordenadas de acuerdo con la lógica y el orden de ejecución
# - [x]  Se ha realizado la tarea 1
#     - [x]  Está presente el procedimiento que puede devolver k clientes similares para un cliente determinado
#     - [x]  Se probó el procedimiento para las cuatro combinaciones propuestas    
#     - [x]  Se respondieron las preguntas sobre la escala/distancia
# - [x]  Se ha realizado la tarea 2
#     - [x]  Se construyó y probó el modelo de clasificación aleatoria para todos los niveles de probabilidad    
#     - [x]  Se construyó y probó el modelo de clasificación kNN tanto para los datos originales como para los escalados. 
#     - [x]  Se calculó la métrica F1.
# - [x]  Se ha realizado la tarea 3
#     - [x]  Se implementó la solución de regresión lineal mediante operaciones matriciales
#     - [x]  Se calculó la RECM para la solución implementada
# - [x]  Se ha realizado la tarea 4
#     - [x]  Se ofuscaron los datos mediante una matriz aleatoria e invertible P    
#     - [x]  Se recuperaron los datos ofuscados y se han mostrado algunos ejemplos    
#     - [x]  Se proporcionó la prueba analítica de que la transformación no afecta a la RECM    
#     - [x]  Se proporcionó la prueba computacional de que la transformación no afecta a la RECM
#     - [x]  Se han sacado conclusiones

# # Apéndices
# 
# ## Apéndice A: Escribir fórmulas en los cuadernos de Jupyter

# Puedes escribir fórmulas en tu Jupyter Notebook utilizando un lenguaje de marcado proporcionado por un sistema de publicación de alta calidad llamado $\LaTeX$ (se pronuncia como "Lah-tech"). Las fórmulas se verán como las de los libros de texto.
# 
# Para incorporar una fórmula a un texto, pon el signo de dólar (\\$) antes y después del texto de la fórmula, por ejemplo: $\frac{1}{2} \times \frac{3}{2} = \frac{3}{4}$ or $y = x^2, x \ge 1$.
# 
# Si una fórmula debe estar en el mismo párrafo, pon el doble signo de dólar (\\$\\$) antes y después del texto de la fórmula, por ejemplo:
# $$
# \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
# $$
# 
# El lenguaje de marcado de [LaTeX](https://es.wikipedia.org/wiki/LaTeX) es muy popular entre las personas que utilizan fórmulas en sus artículos, libros y textos. Puede resultar complicado, pero sus fundamentos son sencillos. Consulta esta [ficha de ayuda](http://tug.ctan.org/info/undergradmath/undergradmath.pdf) (materiales en inglés) de dos páginas para aprender a componer las fórmulas más comunes.

# ## Apéndice B: Propiedades de las matrices

# Las matrices tienen muchas propiedades en cuanto al álgebra lineal. Aquí se enumeran algunas de ellas que pueden ayudarte a la hora de realizar la prueba analítica de este proyecto.

# <table>
# <tr>
# <td>Distributividad</td><td>$A(B+C)=AB+AC$</td>
# </tr>
# <tr>
# <td>No conmutatividad</td><td>$AB \neq BA$</td>
# </tr>
# <tr>
# <td>Propiedad asociativa de la multiplicación</td><td>$(AB)C = A(BC)$</td>
# </tr>
# <tr>
# <td>Propiedad de identidad multiplicativa</td><td>$IA = AI = A$</td>
# </tr>
# <tr>
# <td></td><td>$A^{-1}A = AA^{-1} = I$
# </td>
# </tr>    
# <tr>
# <td></td><td>$(AB)^{-1} = B^{-1}A^{-1}$</td>
# </tr>    
# <tr>
# <td>Reversibilidad de la transposición de un producto de matrices,</td><td>$(AB)^T = B^TA^T$</td>
# </tr>    
# </table>

# # Bibliografía
# ---
# 
# > Los datos curiosos que debes conocer de los seguros. Latino Seguros(23/mar/2022). https://latinoseguros.com.mx/sitio2021/los-datos-curiosos-que-debes-conocer-de-los-seguros-de-vida/
# 
# > Acerca de Compañías de Seguros y Finanzas. DataMéxico(2022). https://datamexico.org/es/profile/industry/insurance-and-surety-companies#:~:text=Acerca%20de%20Compa%C3%B1%C3%ADas%20de%20Seguros%20y%20Fianzas&text=En%20el%20tercer%20trimestre%20de,Fianzas%20registr%C3%B3%201%2C994%20unidades%20econ%C3%B3micas.
# 
# > México, primer lugar en ciberataques en Latinoamérica. Forbes México(2021). https://www.forbes.com.mx/negocios-mexico-primer-lugar-en-ciberataques-en-latinoamerica/
