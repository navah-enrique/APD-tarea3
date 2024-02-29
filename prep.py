"""
Script: prep.py

Este script está diseñado para preparar y limpiar conjuntos de datos para
una tarea de regresión, específicamente para la competencia de Kaggle
"Housing Prices: Advanced Regression Techniques"
(https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

El script incluye funciones para importar datos, manejar valores faltantes,
codificar características categóricas usando OrdinalEncoder, y guardar los
 conjuntos de datos limpios como archivos CSV.

Uso:
1. Descarga los datos de la competencia desde:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/
El archivo "train" será 'raw_train.csv'
y "test" servirá como 'raw_inference.csv'.
2. Coloca los archivos 'raw_train.csv' y 'raw_inference.csv'
en el directorio 'data'.
3. Ejecuta el script con el argparse de los inputs y outputs
para limpiar y preparar los datos.
4. Los conjuntos de datos limpios 'clean_train.csv' y 'clean_inference.csv'
se guardarán en el directorio 'data'.

Funciones:
1. importar_datos():
   - Lee los conjuntos de datos de entrenamiento y de inferencia en bruto.
   - Elimina la columna 'Id' de ambos conjuntos de datos.

2. manejar_valores_faltantes(df, df_test):
   - Maneja valores faltantes en el DataFrame de entrada
   y en el DataFrame para inferencia.
   - Rellena las columnas numéricas con 0 y las columnas categóricas con "N/A".

3. codificar_caracteristicas_ordinarias(df, df_test):
   - Aplica OrdinalEncoder a las características categóricas en el DataFrame
   de entrada y en el de inferencia.
   - Concatena ambos DataFrames, ajusta el codificador en el conjunto de datos
   combinado y transforma ambos DataFrames.

4. guardar_conjuntos_datos_limpios(df, df_test):
   - Guarda los conjuntos de datos de entrenamiento e inferencia limpios
   como 'clean_train.csv' y 'clean_inference.csv'.

Dependencias:
- pandas
- scikit-learn
"""

# %%
# Importamos librerias
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# %%
# Guardamos datos de entrenamiento
# (en la fuente ya vienen separados los archivos de entrenamiento y de test)
# La data se puede descargar desde
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/

# usamos argparse para inputs y outputs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_train')
parser.add_argument('input_inference')
parser.add_argument('output_train')
parser.add_argument('output_inference')
args = parser.parse_args()

# leemos dataset de train
df = pd.read_csv(f'data/{args.input_train}.csv')
# leemos dataset de test
df_test = pd.read_csv(f'data/{args.input_inference}.csv')

df.drop(columns=['Id'], inplace=True)
df_test.drop(columns=['Id'], inplace=True)

# %% [markdown]
# Tenemos 79 variables (quitando ID) y una objetivo (SalePrice).

# %%
df.info()

# %%
df_test.info()

# %%
# Podemos observar que hay varias columnas con datos nulos, pero podemos
# limpiarlas al reemplazarlas por 0 o con N/As dadas las descripciones
# de las columnas
number_columns = df.select_dtypes(include=['float', 'int']).columns.\
   drop('SalePrice')
object_columns = df.select_dtypes(include=['object']).columns

df[number_columns] = df[number_columns].fillna(0)
df[object_columns] = df[object_columns].fillna("N/A")

# Para test
df_test[number_columns] = df_test[number_columns].fillna(0)
df_test[object_columns] = df_test[object_columns].fillna("N/A")

# %%
# Verificamos
df.info()

# %%
df_test.info()

# %%
# Aplicamos transformacion OrdinalEncoder para datos categoricos

# Concatenar df y df_test
combined_df = pd.concat([df, df_test])

# Ajustar el encoder en el dataset combinado
encoder = OrdinalEncoder()
combined_df[object_columns] = encoder.\
   fit_transform(combined_df[object_columns])

# Transformar ambos df y df_test
df[object_columns] = encoder.transform(df[object_columns])
df_test[object_columns] = encoder.transform(df_test[object_columns])

print(df)

# %%
df.head()

# %%
df_test.head()

# %%
# Creamos csvs de la data limpia
df.to_csv(f'data/{args.output_train}.csv', index=False)
df_test.to_csv(f'data/{args.output_inference}.csv', index=False)
