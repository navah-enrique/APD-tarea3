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
import joblib

# %%
#Guardamos dataframes desde los datos ya limpios
df = pd.read_csv('data/clean_train.csv')
df.head()

# %%
#Separamos variables para entrenamiento y prueba
X = df.drop(['SalePrice'],axis=1)
Y = df['SalePrice']
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)

# %%
Test_Y

# %%
#Estandarizamos las variables.

std = StandardScaler()

Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)

Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)

#Mostramos ejemplo para train
Train_X_std.describe()

# %%
#Modelo
#Ejecutamos Regresion Lineal (sin hacer feature selection ni eliminando multicolinealidad)

# Creamos modelo
model = LinearRegression()
model.fit(Train_X_std, Train_Y)

# Guardamos coeficientes
coefficients = model.coef_

coefficients_df1 = pd.DataFrame({
    'Feature': Train_X_std.columns,
    'Coefficient': coefficients
})

# Hacemos predicciones
y_train_pred = model.predict(Train_X_std)
y_test_pred = model.predict(Test_X_std)

# Evaluamos
mse_train = mean_squared_error(Train_Y, y_train_pred)
r2_train = r2_score(Train_Y, y_train_pred)
mse_test = mean_squared_error(Test_Y, y_test_pred)
r2_test = r2_score(Test_Y, y_test_pred)

# Guardamos resultados
results_df = pd.DataFrame({
    'Model': ["RegrLineal"],
    'MSE Train': [mse_train],
    'R-squared Train': [r2_train],
    'MSE Test': [mse_test],
    'R-squared Test': [r2_test]
})

results_df

# %%
#Graficamos resultados de predicción

# Entrenamiento
plt.scatter(Train_Y, y_train_pred)
plt.plot([min(Train_Y), max(Train_Y)], [min(Train_Y), max(Train_Y)], linestyle='--', color='red', linewidth=2)  # 1:1 line
plt.title('Actual vs Predicted para Entrenamiento')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Test
plt.scatter(Test_Y, y_test_pred)
plt.plot([min(Test_Y), max(Test_Y)], [min(Test_Y), max(Test_Y)], linestyle='--', color='red', linewidth=2)  # 1:1 line
plt.title('Actual vs Predicted para Test')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# %%
# Guardamos modelo
filename = 'artifacts/model.joblib'
joblib.dump(model, filename)