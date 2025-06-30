# %% [markdown]
# # El Impacto De Las Condiciones Ambientales
# ## Entregable 1
# 
# ---

# %% [markdown]
# ## Importamos librerías a usar

# %% [markdown]
# [texto del vínculo](https://)

# %%
#Librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from IPython.display import display, HTML
import re
from missingno import matrix, bar, heatmap, dendrogram

# %% [markdown]
# Pandas permite cargar, explorar y manipular datos en estructuras como los DataFrames, que son fundamentales para trabajar con información tabular. numpy se emplea para operaciones numéricas eficientes, como cálculos con arrays y manejo de valores nulos. Para la visualización de datos, se usa matplotlib.pyplot para crear gráficos básicos como líneas, barras o histogramas, mientras que seaborn facilita la generación de gráficos estadísticos más avanzados y estéticos, como mapas de calor o diagramas de caja.
# 
# Para el caso de  coordenadas geográficas, folium es útil para construir mapas interactivos que permiten representar datos espaciales. En cuanto a la preparación y modelado con machine learning, train_test_split de sklearn divide los datos en conjuntos de entrenamiento y prueba, lo que es esencial para validar modelos. StandardScaler se encarga de normalizar variables numéricas, LabelEncoder convierte variables categóricas en valores numéricos, y SimpleImputer permite imputar valores faltantes con distintas estrategias.

# %% [markdown]
# ## Zona de Estudio con GEE

# %% [markdown]
# Utiliza imagenes Sentinel-2 corregido atmosféricamente (COPERNICUS/S2_SR).
# 
# Filtra imágenes con menos del 20% de nubes
# 
# Selecciona la estacion por año (primavera e invierno)
# 
# Primero se debe habilitar una APi de GEE version gratuita, luego se procede a armar un proyecto o carpeta en GEE y se habilita para despues seguir con el codigo en Colab

# %%
# Instalar y cargar Earth Engine
!pip install earthengine-api folium geemap --quiet

import ee
import geemap  # <- IMPORTAR geemap aquí

ee.Authenticate()  # Esto abre un enlace para autorizar tu cuenta
ee.Initialize(project='mentorias-463215')  # Solo después de autenticar

# Punto central: Franja costera del Río de La Plata
punto = ee.Geometry.Point([-58.4464023, -34.5375533])
zona = punto.buffer(1000).bounds()

# Función para obtener una imagen limpia por estación y año
def obtener_imagen(estacion, fecha_ini, fecha_fin):
    coleccion = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(zona) \
        .filterDate(fecha_ini, fecha_fin) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first() \
        .clip(zona)

    vis_params = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}  # RGB
    return coleccion.visualize(**vis_params).set({'title': estacion})

# Crear diccionario con las estaciones
imagenes = {
    'Invierno 2021': obtener_imagen('Invierno 2021', '2021-06-21', '2021-09-21'),
    'Primavera 2021': obtener_imagen('Primavera 2021', '2021-09-22', '2021-12-20'),
    'Invierno 2022': obtener_imagen('Invierno 2022', '2022-06-21', '2022-09-21'),
    'Primavera 2022': obtener_imagen('Primavera 2022', '2022-09-22', '2022-12-20'),
}

# Crear mapa
m = geemap.Map(center=[-34.5375533, -58.4464023], zoom=13)

# Agregar capas al mapa
for nombre, imagen in imagenes.items():
    m.addLayer(imagen, {}, nombre)

m.add_legend(title="Estaciones", labels=list(imagenes.keys()), colors=['#999999']*4)
m  # Mostrar mapa

# %% [markdown]
# Las imágenes satelitales correspondientes a invierno y primavera de los años 2021 y 2022 permiten evidenciar condiciones ambientales superficiales en distintas estaciones y años. Las variaciones en la coloración del agua observadas en las escenas pueden reflejar diferencias en la turbidez, en la presencia de sedimentos en suspensión o incluso en posibles floraciones algales, todos ellos considerados indicadores indirectos del estado de la calidad del agua. Además, el contraste entre áreas ribereñas vegetadas y zonas construidas permite inferir el grado de permeabilidad del entorno y su influencia en los procesos de escurrimiento y arrastre de contaminantes hacia el río.
# 
# Estas imágenes también permiten interpretar procesos ambientales que afectan directamente la calidad del agua. Por ejemplo, un aumento del verdor costero durante la primavera podría asociarse a una mayor actividad agrícola o a un incremento en la escorrentía de fertilizantes, mientras que la presencia de tonalidades marrones u opacas en el cuerpo de agua puede estar relacionada con una mayor carga de sedimentos o materia orgánica en suspensión, resultado de fenómenos como la erosión del suelo, precipitaciones intensas o vertidos cloacales.
# 
# Finalmente, la comparación entre años permite detectar patrones interanuales. Si en una misma estación se observa una mayor degradación visual del agua en un año respecto a otro, es posible vincular esa diferencia con condiciones climáticas particulares (como sequías o lluvias extremas) o con transformaciones en el uso del suelo del entorno inmediato al cauce.

# %% [markdown]
# **NDCI, o Índice de Diferencia Normalizada de Clorofila**
#  es un índice que se utiliza para estimar la concentración de clorofila-a en aguas productivas y turbias, como estuarios, aguas costeras y lagos de agua dulce. Se calcula utilizando bandas espectrales rojas y de borde rojo, que se encuentran en sensores como Sentinel-2.

# %%
import ee
import geemap

ee.Authenticate()
ee.Initialize(project='mentorias-463215')

punto = ee.Geometry.Point([-58.4464023, -34.5375533])
zona = punto.buffer(1000).bounds()

def obtener_imagen(fecha_ini, fecha_fin):
    coleccion = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(zona) \
        .filterDate(fecha_ini, fecha_fin) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first() \
        .clip(zona)
    return coleccion

def calcular_ndci(imagen):
    ndci = imagen.normalizedDifference(['B5', 'B4']).rename('NDCI')
    return ndci

fechas_estaciones = {
    'Invierno 2021': ('2021-06-21', '2021-09-21'),
    'Primavera 2021': ('2021-09-22', '2021-12-20'),
    'Invierno 2022': ('2022-06-21', '2022-09-21'),
    'Primavera 2022': ('2022-09-22', '2022-12-20'),
}

m = geemap.Map(center=[-34.5375533, -58.4464023], zoom=13)

# Parámetros visualización NDCI
ndci_vis = {
    'min': -1,
    'max': 1,
    'palette': ['blue', 'white', 'green', 'darkgreen']
}

for estacion, (fecha_ini, fecha_fin) in fechas_estaciones.items():
    imagen = obtener_imagen(fecha_ini, fecha_fin)
    ndci = calcular_ndci(imagen)
    m.addLayer(ndci, ndci_vis, f'NDCI {estacion}')

colores_rgb = [
    (0, 0, 255),        # azul
    (255, 255, 255),    # blanco
    (0, 128, 0),        # verde
    (0, 100, 0)         # verde oscuro
]

m.add_legend(title="NDCI", labels=['Bajo', 'Medio', 'Alto', 'Muy alto'], colors=colores_rgb)

m

# %% [markdown]
# En Primavera 2021 y Primavera 2022 se observa una mayor intensidad de verde, indicando una mayor concentración de clorofila y, por ende, una mayor presencia de algas en la zona costera. Esto es coherente con que la primavera suele ser temporada de mayor actividad biológica y crecimiento algal debido a mejores condiciones de luz y temperatura.
# 
# En Invierno 2021 y Invierno 2022, la intensidad del verde es menor, mostrando menos concentración de algas. Esto corresponde a la temporada fría, cuando la proliferación algal disminuye por menores temperaturas y condiciones menos favorables para el crecimiento.

# %% [markdown]
# ## Descripcion básica del dataset

# %% [markdown]
# #### Aspectos generales

# %% [markdown]
# Al analizar el dataset Conexiones_Transparentes.csv, se observó primero su dimensión general, es decir, cuántas filas y columnas contiene.Se identificó  613 registros distribuidos en 45 columnas. Esto permitió tener una idea inicial del volumen de datos con los que se va a trabajar. Luego, se revisaron los tipos de datos de cada variable, lo cual permite entender si se trata de valores numéricos, texto, fechas u otro tipo de información, y así anticipar qué tipo de análisis será necesario en cada caso.
# 
# Finalmente, se visualizaron las primeras filas del dataset, lo que sirvió para conocer cómo están estructurados los datos, si los nombres de las columnas son claros, si hay valores atípicos o mal cargados a simple vista, y empezar a identificar patrones o variables relevantes.

# %%
# Configuracion de pandas

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Cargar el dataset

url = "https://raw.githubusercontent.com/MaricelSantos/Mentoria--Diplodatos-2025/main/Conexiones_Transparentes.csv"
df = pd.read_csv(url)

# Descripcion del dataset
print("Dimensiones del dataset:")
print(df.shape)

print("\nTipos de datos:")
print(df.dtypes)

print("\nAlgunas filas del dataset:")
# Acá cambio y elijo mostrar un sampleo más grande. La idea es no ver registros contiguos que puedan
    # tener cierta correlación en la falta de datos
display(df.sample(20, random_state=0))

# %% [markdown]
# En la tabla cada fila representa una columna del DataFrame. Esto significa lo siguiente:
# 
# Column: el nombre de la columna, por ejemplo, orden, sitios, codigo, etc.
# 
# Non-Null Count: cantidad de valores que no están vacíos (no son NaN) en esa columna. Por ejemplo, orden tiene 589 valores no nulos, lo que implica que 24 filas tienen datos faltantes en esa columna (613 - 589 = 24).
# 
# Dtype: el tipo de dato que tiene esa columna:
# 
# - float64: números decimales.
# - object: generalmente texto o una mezcla de tipos (podrían ser números como texto, fechas mal interpretadas, etc.).

# %% [markdown]
# Al revisar los tipos de datos, la mayoría de las columnas están clasificadas como objetos (object). Esto indica que dichas variables o bien contienen datos categóricos o que algunos de sus valores numéricos tienen cierta codificación en formato string (como podría ser el caso de valores faltantes o inexistentes). En este caso, observamos que algunas variables numéricas tienen registros a los que se les asigna una cadena de caracteres en lugar de un valor numérico, correspondiendo estos casos a valores faltantes.
# 
# Se observa también gran cantidad de valores faltantes en las variables añadidas del CENSO y del Programa de Estudios del Conurbano.

# %% [markdown]
# #### Clasificación de las variables

# %% [markdown]
# En las siguientes listas clasificamos las variables según su tipo, observando sus valores e interpretaciones.

# %%
numericas = [
    'tem_agua',
    'tem_aire',
    'od',
    'ph',
    'colif_fecales_ufc_100ml',
    'escher_coli_ufc_100ml',
    'enteroc_ufc_100ml',
    'nitrato_mg_l',
    'nh4_mg_l',
    'p_total_l_mg_l',
    'fosf_ortofos_mg_l',
    'dbo_mg_l',
    'dqo_mg_l',
    'turbiedad_ntu',
    'hidr_deriv_petr_ug_l',
    'cr_total_mg_l',
    'cd_total_mg_l',
    'clorofila_a_ug_l',
    'microcistina_ug_l',
    'ica',
    'Poblacion_partido',
    'Personas_con_cloacas',
]

categoricas = [
    'orden',
    'sitios',
    'codigo',
    'campaña',
    'olores',
    'color',
    'espumas',
    'mat_susp',
    'calidad_de_agua',
    'gobierno_local',
    'sitio',
    'Actividad_principal',
    'Agricultura, ganadería, caza y silvicultura',
    'Pesca explotación de criaderos de peces y granjas piscícolas y servicios conexos',
    'Explotación de minas y canteras',
    'Industria Manufacturera',
    'Electricidad, gas y agua',
    'Construcción',
    'Servicios',
    'fecha',
    'año',
    'latitud',
    'longitud',
]

print("Cantidad de:")
print(f"\tVariables numéricas:   {len(numericas)}")
print(f"\tVariables categóricas: {len(categoricas)}")

# %% [markdown]
# **Notas:** Las variables categóricas fecha y año pueden ser usadas para crear una variable numérica tiempo que mida el transcurso del tiempo desde la primer medición (o desde alguna otra referencia), es decir, tenemos en estas codificadas información numérica, pero las clasificamos como categóricas.

# %% [markdown]
# Veamos la cantidad de valores únicos de cada variable.

# %%
# Crear DataFrame para clasificar variables y mostrar en tabla
variables = pd.DataFrame({
    'Variable': df.columns,
    'Tipo': ['Numerica' if col in numericas else 'Categorica' for col in df.columns],
    'Cantidad de valores únicos': [df[col].nunique() for col in df.columns]
})

display(variables)



# %% [markdown]
# #### Calidad de los datos

# %% [markdown]
# Vemos los porcentajes de valores no numéricos, nulos y vacíos -carácter ""- en las columnas numéricas. 
# Además, vemos la cantidad de valores de forma forma '<{n}', con n un número. Estos últimos no son contabilizados
# dentro de los valores no numéricos en esta sección.

# %%
# Patrón para valores como <n
patron = re.compile(r'^<\d+\.?\d*$')  # Acepta <3, <3.5, etc.

print("Porcentaje de:")
print("Columna\t\t\t\tCeros\tNo numéricos\tCon forma '<x'\tNulos\tVacíos")
print("______________________________________________________________________________________")
for col in numericas:
    # Intentar convertir a numérico
    valores_numericos = pd.to_numeric(df[col], errors='coerce')
    no_numericos = df[col][valores_numericos.isna() & df[col].notna()]
    
    # Contar cantidad de valores ...
    cantidad_no_numericos = len(no_numericos) # no numéricos
    cantidad_ceros = (df[col] == 0).sum() # iguales a cero
    vacios = df[col].str.strip().eq("").sum() # Vacíos -carácter "" o algún tipo de espacio-
    nulos = df[col].isnull().sum() # Nulos
    if cantidad_no_numericos != 0:
        valores_menor_que = no_numericos.apply(lambda x: bool(patron.match(str(x)))).sum() # Valores de forma '<x'
    else:
        valores_menor_que = 0
    
    # Cálculo de porcentajes
    total = len(df)
    porc_ceros = cantidad_ceros*100/total
    porc_no_numericos = (cantidad_no_numericos-valores_menor_que)*100/total
    porc_menores_que = valores_menor_que*100/total
    porc_nulos = nulos*100/total
    porc_vacios = vacios*100/total
    print(f"{col:<{25}}\t{porc_ceros:5.2f}\t   {porc_no_numericos:5.2f}\t    {porc_menores_que:5.2f}\t{porc_nulos:5.2f}\t{porc_vacios:5.2f}")

# %% [markdown]
# Observamos ahora los porcentajes de valores nulos y vacíos de las variables categóricas

# %%
print("Porcentaje de:")
print("Columna\t\t\t\tNulos\tVacíos")
print("______________________________________________________________________________________")
for col in categoricas:
    # Convertir temporalmente a string
    columna_str = df[col].astype(str)
    
    # Contar cantidad de valores ...
    vacios = columna_str.str.strip().eq("").sum() # Vacíos -carácter "" o algún tipo de espacio-
    nulos = df[col].isnull().sum() # Nulos
    
    # Cálculo de porcentajes
    total = len(df)
    porc_nulos = nulos*100/total
    porc_vacios = vacios*100/total
    print(f"{col[:25]:<{25}}\t{porc_nulos:5.2f}\t{porc_vacios:5.2f}")

# %% [markdown]
# Visualizamos brevemente cómo se relacionan los datos faltantes de las variables, observando una muestra de todo el dataset para mayor claridad

# %%
matrix(df.sample(250))
heatmap(df.sample(250))

# %% [markdown]
# Se observa una fuerte correlación entre datos faltantes. En particular, se ve que hay fuerte correlación entre los datos faltantes de las variables provistas por el Ministerio de Ambiente y Desarrollo Sostenible y entre los datos faltantes de las variables del CENSO y del Programa de Estudios del Conurbano.

# %% [markdown]
# #### Observamos la variable objetivo: ICA

# %%
print("Resumen de descripción de la variable objetivo ICA")

col = 'ica'

# Total de valores
total = df[col].shape[0]

# Valores nulos
nulos = df[col].isnull().sum()

# Valores vacíos o espacios en blanco
faltantes = df[col].astype(str).str.strip().replace('', pd.NA).isna().sum()

# Detectar valores mal cargados (no convertibles a numérico, excluyendo nulos)
valores_convertidos = pd.to_numeric(df[col], errors='coerce')
mal_cargados_mask = valores_convertidos.isna() & (~df[col].isnull())
mal_cargados = df.loc[mal_cargados_mask, col]

# Cantidad de valores mal cargados
cant_mal_cargados = mal_cargados.shape[0]

# Crear tabla resumen
resumen = pd.DataFrame({
    'Descripción': ['Total de valores', 'Valores nulos (NaN)', 'Valores vacíos/faltantes', 'Valores mal cargados'],
    'Cantidad': [total, nulos, faltantes, cant_mal_cargados]
})

display(resumen)

# Mostrar tabla con valores mal cargados
print(f"\nValores mal cargados en '{col}':")
display(mal_cargados.reset_index(drop=True))

# %% [markdown]
# Observamos que los valores mal cargados de ICA se corresponden a valores falantes.

# %% [markdown]
# ## Analisis exploratorio

# %% [markdown]
# La información general del dataset muestra un total de 613 registros distribuidos en 45 columnas. Sin embargo, no todas las columnas tienen la misma cantidad de datos completos. Algunas presentan valores faltantes, como es el caso de tem_aire , lo que indica que ciertas observaciones no fueron medidas o registradas en todas las campañas o estaciones.
# 
# Además, se nota que la mayoría de las columnas están tipadas como object, incluso aquellas que deberían contener valores numéricos, como od, nitrato_mg_l, entre otras. Esto sugiere que probablemente haya que hacer una conversión de tipos para poder analizarlas correctamente, ya sea por el uso de comas como separadores decimales o por la presencia de caracteres no numéricos.
# 
# Esta revisión general  permite detectar desde el inicio qué variables requieren limpieza o transformación antes de cualquier análisis o modelado más avanzado. También da una idea del nivel de completitud de los datos y de los posibles desafíos que habrá que enfrentar en el preprocesamiento.

# %% [markdown]
# Previo a continuar con un análisis más detallado de los datos, se observa que:
# - En muchas variabes numéricas tenemos valores dados por cadenas de caracteres como 'no detectados'. Estos corresponden a valores faltantes, por lo que se transformarán en NaN
# - Por otro lado, tenemos valores en variables numéricas de la forma '<100', estos corresponden a casos en los que se detectó la sustancia medida, pero no puede determinarse su cantidad al ser esta menor al LOQ. A falta de un mejor criterio, de momento, estos valores se imputarán como 0. Como no hay ceros en las variables numéricas, es seguro haceer esto.
# - En la variable fecha tenemos strings como "no se midió" o números como 45623, que no corresponden a fechas válidas. Se interpreta como causa la conversión automática que hace Excel si no se le especifica el formato fecha Se aplica una función de python para detectar dichos casos y convertirlos a formato dd/mm/yyyy. Aquellos valores que no tengan la forma dd/mm/aaaa serán convertidos a null.
# - En la variable campaña, tenemos algunos valores con mayúsculas y otros con minúsculas. Se pasan todos a minúscula. Se imputan los valores mal imputados como None.
# - En algunas variables categóricas como olores, color, espumas y mat_susp, en las que los valores admitidos son Ausencia o Presencia, tenemos otros valores asignados, como por ejemplo 'en obra', que corresponden a valores faltantes. Estos se imputan como None. Además, uniformizamos los valores para tener sólo "Ausente" o "Presente", puesto que también están presentes las opciones 'Ausencia'/'Presencia' y sus versiones en minúscula.
# - Algo similar ocurre en calidad del auga. Se procede de forma similar.
# 
# Antes de proseguir con un análisis más detallado, realizamos algunas correcciones sobre el dataset. Luego, durante el análisis, hacemos una comparación entre el dataset original y el obtenido luego de corregir los puntos anteriores.

# %% [markdown]
# #### Corrección dataset

# %% [markdown]
# Se crea función para identificar casos de fechas del estilo '45623' y se convierten al formato dd/mm/yyyy

# %%
from datetime import datetime, timedelta

df1 = df.copy(deep=True)
# Function to convert Excel serial number to date string
def convert_excel_serial_date(value):
    # print(type(value))
    if pd.isna(value):
        return value
    elif value == "31/10/0202":
        return "31/10/2022"
    elif "/" not in value and "no" not in value:
        int(value)
        base_date = datetime(1899, 12, 30)
        after_value = (base_date + timedelta(days=int(value))).strftime("%d/%m/%Y")
        print(f"Converting {value} to {after_value}")
        return after_value
    return value

# Apply fix
df1["fecha"] = df1["fecha"].apply(convert_excel_serial_date)

# %%
# Seteamos a NaT los valores de fecha incorrectos para contabilizar correctamente la cantidad de valores nulos
    # en la columna fecha
# Ejemplo de DataFrame
# 1. Convertir a datetime con formato dd/mm/aaaa, forzando errores a NaN
df2 = df1.copy(deep=True)
df2['fecha'] = pd.to_datetime(df2['fecha'], format='%d/%m/%Y', errors='coerce')

# %%
# Reemplazamos los valores no numéricos y de la forma '<x'
df3 = df2.copy(deep=True)

for col in numericas:
    # Convertir valores '<x' a '0'
    valores = df3[col].astype(str)  # Convertir todo a string para comparar
    mascara_menor_que = valores.apply(lambda x: bool(patron.match(x)))
    df3.loc[mascara_menor_que, col] = '0'
    
    # Convertir la columna a numérico (los strings no convertibles irán a NaN)
    df3[col] = pd.to_numeric(df3[col], errors='coerce')

# %%
# Pasamos todos los valores de campaña a minúscula
df4 = df3.copy(deep=True)
df4['campaña'] = df4['campaña'].str.lower().where(df4['campaña'].notna(), None)

print(f"Antes de la transformación:\n {df3['campaña'].unique()}")
print(f"Después de la transformación:\n {df4['campaña'].unique()}")

# %%
'''
Se convierten a None aquellos valores mal asignados en la columna campaña
'''
print(f"Antes de la transformación:\n {df4['campaña'].unique()}")

col = 'campaña'
patron = re.compile(r'^(verano|otoño|invierno|primavera)', flags=re.IGNORECASE)

df4[col] = df4[col].where(
    df4[col].astype(str).str.contains(patron, na=False)
)
print(f"Después de la transformación:\n {df4['campaña'].unique()}")

# %%
'''
En aquellas columnas que sólo admiten valores Asencia/Presencia, imputamos como None aquellos que no
correspondan a alguna de las correspondientes categorías. Además, uniformamos los valores a 'Ausente'/'Presente'
'''
# Copia del DataFrame
df5 = df4.copy(deep=True)

# Columnas a procesar
cols_a_p = ['olores', 'color', 'espumas', 'mat_susp']

# Patrón para detectar "ausencia/presencia" (ignorando mayúsculas y variantes)
patron = re.compile(r'^(ausen|presen)', flags=re.IGNORECASE)

for col in cols_a_p:
    print(f"Columna: {col}")
    # Extraer la raíz (primeras 5-6 letras) del valor
    raiz_valor = df5[col].astype(str).str.extract(patron)[0]
    
    # Mapear a "Ausente" o "Presente" (y NaN si no coincide)
    df5[col] = raiz_valor.str.lower().map(
        {
            'ausen': 'Ausente',
            'presen': 'Presente'
        }
    )
    print(f"\tAntes:\n\t\t{df4[col].unique()}")
    print(f"\tDespués:\n\t\t{df5[col].unique()}")

# %%
# Se imputan como None aquellos valores no válidos en calidad_de_agua
df6 = df5.copy()

opciones_validas_calidad = [
    'Apta',
    'Levemente deteriorada',
    'Deteriorada',
    'Muy deteriorada',
    'Extremadamente deteriorada'
]
df6['calidad_de_agua'] = df6['calidad_de_agua'].where(df6['calidad_de_agua'].isin(opciones_validas_calidad))

print(f"Antes de la transformación:\n {df5['calidad_de_agua'].unique()}")
print(f"Después de la transformación:\n {df6['calidad_de_agua'].unique()}")

# %%
"""
Por último, observo que los valores de las variables Poblacion_partido y Personas_con_cloacas me quedaron como 
punto flotantes, pues los puntos que separaban cada 3 cifras en español fueron interpretados como comas.
Arreglo este problema
"""
df7 = df6.copy()
for col in ['Poblacion_partido', 'Personas_con_cloacas']:
    max_decimales = df7[col].dropna().apply(lambda x: len(str(x).split('.')[1])).max()
    # Convertir multiplicando por 10^max_decimales
    df7[col] = (df7[col] * (10 ** max_decimales)).astype('Int64')

# %%
display(df7.sample(20, random_state=0))

# %%
# Comparamos info general del dataset original y el corregido
print("\tColumna\t\t\t\t\tTipo Orig\tNulos Orig (%)\tTipo Corr.\tNulos Corr. (%)")
print("_______________________________________________________________________________________________________________")
print("")
total = len(df)
for col in df.columns:
    # Tipo de datos de la columna
    tipo_dato_original = df[col].dtype
    tipo_dato_corregido = df7[col].dtype
    # Cantidad de datos no nulos
    no_nulos_orginal = df[col].count()
    no_nulos_corregido = df7[col].count()
    # Porcentaje de datos nulos
    nulos_orig = 100 - no_nulos_orginal*100/total
    nulos_corr = 100 - no_nulos_corregido*100/total
    print(f"\t{col[:26]:<{25}}\t\t  {str(tipo_dato_original):<{7}}\t  {nulos_orig:5.2f}\t\t {str(tipo_dato_corregido)[:8]:<{7}}\t   {nulos_corr:5.2f}")

# %% [markdown]
# **Procesamiento de datos**

# %% [markdown]
# Indica cuántos registros están completamente duplicados, es decir, que repiten todos los valores en todas sus columnas. Este paso permite asegurarse de que no haya información repetida que pueda sesgar los análisis o inflar resultados sin aportar datos nuevos. Saber si existen duplicados también ayuda a decidir si es necesario limpiar el dataset antes de seguir trabajando.

# %%
# Datos duplicados
print("\nCantidad de datos duplicados:")
print(df.duplicated().sum())

# %% [markdown]
# **Identificacion de datos duplicados**

# %% [markdown]
# Este analisis permite identificar si existen registros exactamente repetidos dentro del dataset. Primero se confirmó cuántas filas completas están duplicadas, lo que indica que ciertos registros fueron cargados más de una vez sin ningún cambio en ninguna de sus columnas. Estas filas duplicadas fueron mostradas en una tabla para poder inspeccionarlas visualmente y decidir si conviene eliminarlas en el preprocesamiento.
# 
# Luego se revisó cada columna por separado para detectar si contiene valores repetidos, aunque no necesariamente en filas idénticas. Esto ayuda a identificar, por ejemplo, si hay valores que se repiten demasiado en campos que deberían ser únicos (como una estación de monitoreo o una fecha) o si hay agrupamientos naturales que podrían aprovecharse durante el análisis.

# %%
# Filas duplicadas
print("\nCantidad de filas duplicadas:")
print(df.duplicated().sum())

print("\nFilas duplicadas:")
duplicadas = df[df.duplicated()]
display(duplicadas)

# Valores duplicados de cada columna
print("\nColumnas con valores duplicados:")
for col in df.columns:
    valores_duplicados = df[col][df[col].duplicated()].unique()
    if len(valores_duplicados) > 0:
        print(f"\nColumna: {col}")
        print(f"Valores duplicados: {valores_duplicados}")

# %% [markdown]
# **Valores faltantes - Datos vacios o con NaN**

# %% [markdown]
# Este código analiza un DataFrame y genera una tabla resumen que muestra cuántos datos faltantes tiene cada columna. Lo que hace es calcular cuántos valores son NaN, cuántos son cadenas vacías "", y luego suma ambos para obtener el total de faltantes. A partir de eso, también calcula el porcentaje que representan esos valores faltantes en relación al total de filas del DataFrame.
# 
# El resultado es una tabla con cuatro columnas: una con la cantidad de valores NaN, otra con la cantidad de celdas vacías como texto, otra con el total combinado de valores faltantes, y una última con el porcentaje de datos incompletos. Esta tabla se muestra con un estilo visual que aplica una escala de color que va del verde al azul para resaltar visualmente las columnas con más valores faltantes. Cuanto más alto es el valor, más oscuro es el color de fondo.
# 
# Además, la columna que muestra el porcentaje de faltantes aplica un formato especial: si ese porcentaje supera el 50%, el valor aparece resaltado en rojo y en negrita. Esto permite identificar rápidamente qué variables están críticamente incompletas y podrían requerir limpieza o eliminación.
# 

# %%
# Calcular valores faltantes
nan_counts = df.isna().sum()
empty_counts = (df == "").sum()
total_faltantes = df.isna() | (df == "")
total_missing_counts = total_faltantes.sum()
porcentaje = (total_missing_counts / len(df) * 100)

# Detectar valores mal cargados en variables numéricas
mal_cargados_counts = {}
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Para columnas que no son numéricas, intentar convertir y contar errores
for col in df.columns:
    # Solo columnas no numéricas
    if col not in num_cols:
        converted = pd.to_numeric(df[col], errors='coerce')
        mal_cargados_counts[col] = converted.isna().sum() - nan_counts[col]
    else:
        mal_cargados_counts[col] = 0  # Asumimos que numéricas están correctas o NaN ya contados

mal_cargados_counts = pd.Series(mal_cargados_counts)

# Crear resumen completo
summary = pd.DataFrame({
    "Valores NaN": nan_counts,
    "Vacíos ('')": empty_counts,
    "Total Faltantes": total_missing_counts,
    "Mal cargados": mal_cargados_counts,
    "Porcentaje (%)": porcentaje
})

# Función para resaltar en rojo si el porcentaje es crítico (> 50%)
def resaltar_porcentaje(col):
    return ['color: red; font-weight: bold' if v > 50 else '' for v in col]

# Aplicar estilos
summary.style\
    .background_gradient(cmap="BuGn")\
    .format({"Porcentaje (%)": "{:.2f}"})\
    .apply(resaltar_porcentaje, subset=["Porcentaje (%)"])\
    .set_caption("Resumen de Datos Faltantes e Inconsistentes")\
    .set_properties(**{'text-align': 'center'})

# %% [markdown]
# **Resumen estadistico**

# %% [markdown]
# Al ejecutar el resumen estadístico del dataset, el objetivo principal es obtener una visión general cuantitativa y cualitativa de cada variable. Esto incluye información como: la cantidad de datos disponibles, los valores únicos, la moda, la media, la mediana, los cuartiles, el mínimo y el máximo, así como la desviación estándar para las variables numéricas.
# 
# Este análisis permite identificar rápidamente la distribución y características principales de los datos, detectar posibles valores atípicos, entender la diversidad o repetición en variables categóricas, y evaluar la calidad y completitud de la información.

# %%
# Resumen estadistico
resumen = df.describe(include='all')

# Renombrar índices al español
resumen.rename(index={
    'count': 'Cantidad de datos',
    'unique': 'Valores únicos',
    'top': 'Valor más frecuente',
    'freq': 'Frecuencia',
    'mean': 'Promedio',
    'std': 'Desviación estándar',
    'min': 'Mínimo',
    '25%': 'Percentil 25',
    '50%': 'Mediana (Percentil 50)',
    '75%': 'Percentil 75',
    'max': 'Máximo'
}, inplace=True)

# Función para alternar colores en filas
def estilo_tabla(s):
    colores = ['#f9f9f9', '#e0f7fa']
    return ['background-color: {}'.format(colores[i % 2]) for i in range(len(s))]

# Aplicar estilo y mostrar
styled_resumen = resumen.style.apply(estilo_tabla, axis=0)\
                             .set_properties(**{'font-weight': 'bold'}, subset=pd.IndexSlice[['Cantidad de datos', 'Valores únicos', 'Frecuencia', 'Desviación estándar'], :])\
                             .set_table_styles([{
                                'selector': 'th',
                                'props': [('background-color', '#00796b'), ('color', 'white'), ('font-weight', 'bold')]
                             }])

print("\nResumen estadístico")
display(styled_resumen)

# %% [markdown]
# Al observar el resumen estadístico del dataset, se puede observar que algunas columnas tienen una cantidad menor de datos registrados, lo que indica la presencia de valores faltantes. Por ejemplo, mientras que variables como el código o la fecha cuentan con casi todos los datos, otras como las mediciones químicas o físicas tienen menos registros completos.
# 
# Se destaca que algunas columnas contienen valores categóricos, como el nombre de los sitios, la campaña o la estación del año, donde el valor más frecuente se repite muchas veces, reflejando condiciones comunes o datos estándar (como “invierno” o “Ausencia” en ciertas mediciones).
# 
# En las columnas numéricas, el promedio, la desviación estándar y los percentiles muestran la distribución de los datos cuando están presentes, pero muchos valores son nulos, lo que implica que será necesario realizar un tratamiento especial para estos datos faltantes antes de hacer un análisis más profundo.

# %% [markdown]
# **Distribucion de las variables**

# %% [markdown]
# Los gráfico de histogramas muestra cómo se distribuyen los valores de cada variable numérica en el dataset. Permite visualizar si los datos están concentrados alrededor de ciertos rangos, si presentan sesgos hacia un extremo, o si existen valores atípicos o outliers. Además, ayuda a identificar la forma de la distribución por ejemplo, si es normal, sesgada o multimodal.

# %%
# Seleccionar columnas numéricas excepto 'orden'
num_cols = [col for col in df.select_dtypes(include=np.number).columns if col.lower() != 'orden']

# Graficar histogramas solo para esas columnas
df[num_cols].hist(bins=30, figsize=(18, 12))
plt.suptitle("Distribución de las variables (excluyendo 'orden')", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# En Agricultura, ganadería, caza y silvicultura, las categorías 0 y 4 son las más frecuentes, con una presencia destacada también de las categorías 1 y 3. Esto indica que ciertas actividades o subsectores dentro de esta área son más comunes en los datos.
# 
# En el sector de Pesca y explotación de criaderos de peces y granjas piscícolas, la categoría 4 tiene la mayor frecuencia, seguida por las categorías 2, 3 y 5, mostrando una concentración significativa en estas actividades específicas.
# 
# Para la Explotación de minas y canteras, la categoría 0 es la que domina ampliamente, con la categoría 4 también teniendo un peso considerable. Esto refleja una concentración en ciertas actividades mineras o extractivas dentro del dataset.
# 
# En la Industria manufacturera, la categoría 4 es claramente la más frecuente, seguida por las categorías 2, 6 y 7, lo que señala que una parte importante de la industria representada se centra en esos subsectores.
# 
# En el sector de Electricidad, gas y agua, la categoría 5 destaca por su alta frecuencia, con presencia moderada de las categorías 1, 3 y 6, reflejando la distribución de actividades en servicios públicos.
# 
# En Construcción, la categoría 3 tiene la mayor frecuencia, seguida por las categorías 1, 5 y 6, mostrando una variedad de subsectores con una concentración significativa en ciertos tipos de construcción.
# 
# Finalmente, en Servicios, la categoría 4 es la más común, con las categorías 1, 3 y 6 también presentes en proporciones relevantes, indicando la diversidad dentro del sector servicios y ciertos subgrupos predominantes.
# 
# 
# 
# 

# %% [markdown]
# **Datos atipicos**

# %% [markdown]
# Se excluyo la columna orden

# %%
# Seleccionar columnas numéricas excluyendo 'orden'
num_cols = [col for col in df.select_dtypes(include=np.number).columns if col.lower() != 'orden']

plt.figure(figsize=(18, 8))
sns.boxplot(data=df[num_cols])
plt.title("Detección de outliers en variables numéricas")

# Modificar etiquetas insertando saltos de línea
labels_wrapped = [label.replace(" ", "\n").replace(",", "\n") for label in num_cols]

plt.xticks(ticks=range(len(labels_wrapped)), labels=labels_wrapped, rotation=0, ha='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# El gráfico de cajas muestra la distribución de varias variables numéricas del dataset y permite identificar la presencia de outliers. En la mayoría de las variables relacionadas con sectores como agricultura, pesca, minería, industria manufacturera, electricidad, construcción y servicios, las cajas y los bigotes son relativamente compactos, lo que indica una distribución controlada y poca dispersión extrema.
# 
# Sin embargo, la variable "orden" presentaba una mayor dispersión y una caja más grande, sugiriendo que podría contener valores extremos o outliers que se alejan significativamente del resto de los datos. Lo cual se excluyo de la visualizacion y analisis.
# 
# Para las otras variables, no se observan puntos fuera de los bigotes, lo que indica que no hay outliers evidentes o que estos son muy pocos y no se destacan claramente en el gráfico.
# Algunos sectores como Industria Manufacturera y Pesca tienen medianas y rangos más altos, indicando que suelen tener valores mayores.
# Otros sectores como Agricultura o Explotación de minas y canteras tienen medianas más bajas y rangos más cortos.

# %% [markdown]
# **Correlacion de variables**

# %% [markdown]
# Se muestra una matriz de correlación entre las variables numéricas del dataset, permitiendo visualizar qué tan relacionadas están entre sí. Los valores anotados indican la fuerza y dirección de la relación: valores cercanos a 1 o -1 muestran correlaciones fuertes positivas o negativas, respectivamente, mientras que valores cercanos a 0 indican poca o ninguna correlación.
# 

# %%
# Correlacion entre variables
plt.figure(figsize=(16, 10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de correlación")
plt.xticks(ticks=range(len(labels_wrapped)), labels=labels_wrapped, rotation=0)
plt.yticks(rotation=0)  # etiquetas verticales
plt.tight_layout()
plt.show()

# %% [markdown]
# En esta matriz se observa que algunas variables, como “Industria Manufacturera” y “Electricidad, gas y agua”, tienen una correlación positiva moderada (0.57), lo que indica que estas dos actividades tienden a aumentar o disminuir juntas. Por otro lado, “Industria Manufacturera” y “Servicios” muestran una correlación negativa fuerte (-0.73), lo que sugiere que cuando una variable aumenta, la otra tiende a disminuir.
# 
# También destaca que “Explotación de minas y canteras” tiene una correlación positiva moderada con “Servicios” (0.43) y una correlación negativa con “Industria Manufacturera” (-0.52), lo que puede reflejar diferencias en la dinámica económica entre estos sectores.

# %% [markdown]
# **Correlacion entre Gobierno local y codigo**
# Tabla de contingencia muestra cómo se distribuyen las observaciones según las categorías combinadas

# %%
from scipy.stats import chi2_contingency

# Crear tabla de contingencia
contingency_table = pd.crosstab(df['codigo'], df['gobierno_local'])

print("Tabla de contingencia entre 'codigo' y 'gobierno_local':")
display(contingency_table)

# Calcular Chi-cuadrado
chi2, p, dof, ex = chi2_contingency(contingency_table)

# Calcular coeficiente de Cramér
n = contingency_table.sum().sum()
cramer_v = np.sqrt(chi2 / (n * (min(contingency_table.shape)-1)))

print(f"\nEstadístico Chi-cuadrado: {chi2:.4f}")
print(f"Valor p: {p:.4f}")
print(f"Grados de libertad: {dof}")
print(f"Coeficiente de Cramér: {cramer_v:.4f}")

# %% [markdown]
# Existe una asociación estadísticamente significativa y muy fuerte entre las variables codigo y gobierno_local. Esto significa que cada código está asociado de manera casi exclusiva con un gobierno local específico, lo cual es coherente si los códigos representan sitios o unidades que pertenecen a un solo gobierno local.

# %% [markdown]
# CONCLUSION

# %% [markdown]
# El dataset contiene un total de 45 variables o features y 613 entradas o registros. Estas variables incluyen tanto datos numéricos como categóricos.
# 
# En cuanto al tipo de datos, muchas columnas están clasificadas como objetos (object), lo que indica que son categóricas o texto. Sin embargo, algunas variables que deberían ser numéricas están almacenadas como texto, por lo que será necesario realizar conversiones para trabajar con datos cuantitativos continuos o discretos. También existen variables categóricas y posiblemente binarias dentro del conjunto.
# 
# Se detectaron algunas filas duplicadas, aunque en cantidad limitada, y también valores repetidos dentro de columnas, lo cual es común en datos categóricos pero debe ser evaluado en variables numéricas para evitar redundancias.
# 
# Respecto a los valores faltantes, varias columnas presentan datos incompletos, lo que implica que se deberá decidir cómo imputar esos valores o si es necesario eliminar registros o variables con muchos datos faltantes para mantener la calidad del análisis.
# 
# El análisis de distribución mediante histogramas y boxplots reveló que algunas variables presentan outliers o valores atípicos, especialmente la variable “orden”, mientras que otras variables muestran distribuciones más compactas y homogéneas.
# 
# La matriz de correlación indicó relaciones positivas y negativas entre ciertas variables, por ejemplo, una correlación positiva moderada entre “Industria Manufacturera” y “Electricidad, gas y agua”, y una correlación negativa fuerte entre “Industria Manufacturera” y “Servicios”. Estas relaciones ayudan a entender la interacción entre variables, identificar redundancias y seleccionar aquellas más relevantes para modelar o interpretar.
# 
# Respecto a las unidades, es importante revisar y estandarizar aquellas variables numéricas que puedan estar expresadas en diferentes escalas o formatos, para evitar sesgos en el análisis o modelado.


