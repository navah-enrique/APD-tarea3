# %%
#Importamos librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#import statsmodels.api as sm
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import Lasso
#from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OrdinalEncoder

# %%
#Guardamos datos de entrenamiento (en la fuente ya vienen separados los archivos de entrenamiento y de test)
# La data se puede descargar desde https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/
df = pd.read_csv('data/raw_train.csv')
df.drop(columns=['Id'], inplace=True)
df.head()

#leemos dataset de test
df_test = pd.read_csv('data/raw_inference.csv')
df_test.drop(columns=['Id'], inplace=True)

# %% [markdown]
# Tenemos 79 variables (quitando ID) y una objetivo (SalePrice).

# %%
df.info()

# %%
df_test.info()

# %%
#Podemos observar que hay varias columnas con datos nulos, pero podemos limpiarlas al reemplazarlas
#por 0 o con N/As dadas las descripciones de las columnas
number_columns = df.select_dtypes(include=['float','int']).columns.drop('SalePrice')
object_columns = df.select_dtypes(include=['object']).columns

df[number_columns] = df[number_columns].fillna(0)
df[object_columns] = df[object_columns].fillna("N/A")

#Para test
df_test[number_columns] = df_test[number_columns].fillna(0)
df_test[object_columns] = df_test[object_columns].fillna("N/A")

# %%
#Verificamos
df.info()

# %%
df_test.info()

# %%
#Aplicamos transformacion OrdinalEncoder para datos categoricos

# Concatenar df y df_test
combined_df = pd.concat([df, df_test])

# Ajustar el encoder en el dataset combinado
encoder = OrdinalEncoder()
combined_df[object_columns] = encoder.fit_transform(combined_df[object_columns])

# Transformar ambos df y df_test
df[object_columns] = encoder.transform(df[object_columns])
df_test[object_columns] = encoder.transform(df_test[object_columns])

print(df)

# %%
df.head()

# %%
df_test.head()

# %%
#Creamos csvs de la data limpia
df.to_csv('data/clean_train.csv', index=False)
df_test.to_csv('data/clean_inference.csv', index=False)


