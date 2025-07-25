## Check list post reunión con Profe - 08/07/2025
- [ ] Hacer cambios a partir del dataframe, deberíamos exportarlo
- [ ] Eliminar latitud, longitud, fecha 
- [ ] Eliminar las columnas que superen el 50% de datos faltantes 
- [ ] Los valores <x, cuyo porcentaje sea menor al 10% se podría tratar como faltante
- [ ] ver que pasa con la distribución de los valores <x, de ahí ver si dejarlo o eliminarlo
- [ ] No olvidarse de poner nuestros nombre al principio del tp porque fue una corrección

## Check list post reunión con Profe - 24/06/2025
- [ ] Usar un trabajo de  base y agregar algunas cosas de Edu al principio como:
  - [ ] La parte de valores no numéricos: no se midio, <x (valor), cantidad de 0 
  - [ ] Agregarle porcentaje en vez de valores absolutos y eliminar aquellos que superen un determinado %, por ejemplo 60%
- [ ] Agregar al tp base Clasificación de categóricas y numérica ( Diferenciación de Discretas y continuas)
- [x] Transformación  o modificación de:
  - Columna fecha: De 50 valores mal cargados, 24 son nulos y el resto es por: no se midió, no se muestreo, o tipeo 31/10/0202 (fueron 2), patrón 45623
- [ ] Ver y determinar cuantos son :
 - [ ] Los valores < x (valor) : ver si agregarle 0, sacarle el signo o tomar un rango o limite
 - [ ] Decidir en los casos de que sea <0.005 y <10 ver esos casos normativa o parámetros permitidos
- [ ] Crear: una matriz de correlación de algunas variables (categóricas y numéricas) y ver cuales agrupar

## Check list post reunión con Profe - 17/06/2025
- [x] Nos falta identificar cuales son categóricas y cuales numéricas
- [x] Eliminar la categoría orden en el grafico box plot
- [x] Utilizar el gráfico de correlación Código y Gobierno Local
- [ ] Utilizar la columna Código que es más completa. El código hace referencia al municipio (ayuda a completar esos  valores NaN) 
- [x] Analizar la columna ICA
- [ ] A traves de la matriz de correlación de variables surge la posibilidad de eliminar por ejemplo servicios 
- [ ] Lo que dice no se midió en toda una fila, se puede eliminar 
- [x] Evaluar posibilidades de que hacer con las filas que tienen valores por ejemplo <30 o <2.0, agrupar o no.
- [x] Ver aquellas filas que contienen 0 si son óptimas o no con respecto a la categoría

# Comandos útiles
### Convert .ipynb → .py (Notebook to Python script)
- `jupyter nbconvert --to script your_notebook.ipynb`

### Convert .py → .ipynb (Python script to Notebook)
- pip install jupytext
- jupytext --to notebook your_script.py
To sync both files:
- jupytext --set-formats ipynb,py your_notebook.ipynb
