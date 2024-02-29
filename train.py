"""
Script: train.py

Este script se encarga de entrenar un modelo de regresión lineal
utilizando datos limpios preparados con el script 'prep.py'.
Incluye la separación de conjuntos de entrenamiento y prueba,
estandarización de variables, entrenamiento del modelo y evaluación
de su rendimiento en los conjuntos de entrenamiento y prueba.
Además, se guardan los resultados y el modelo entrenado.

Uso:
1. Se requiere haber ejecutado 'prep.py' previamente para obtener los
conjuntos de datos limpios.
2. Ejecutar este script para entrenar el modelo de regresión lineal.
3. Los resultados del rendimiento del modelo se mostrarán en la consola
y se guardarán en results_df.
4. El modelo entrenado y el objeto StandardScaler se guardarán en el
directorio 'artifacts' como 'model.joblib' y 'std.joblib' respectivamente.

Funciones:
1. cargar_datos_limpios():
   - Lee el conjunto de datos limpio 'clean_train.csv'.

2. separar_entrenamiento_prueba(X, Y):
   - Separa las variables independientes (X) y la variable dependiente (Y)
     en conjuntos de entrenamiento y prueba.

3. estandarizar_variables(train_X, test_X):
   - Estandariza las variables utilizando StandardScaler.

4. entrenar_modelo(train_X_std, train_Y):
   - Entrena un modelo de regresión lineal en las variables estandarizadas
     y la variable dependiente de entrenamiento.

5. evaluar_modelo(model, train_X_std, test_X_std, train_Y, test_Y):
   - Evalúa el rendimiento del modelo en los conjuntos de entrenamiento
   y prueba.

6. graficar_predicciones(train_Y, y_train_pred, test_Y, y_test_pred):
   - Grafica las predicciones del modelo en los conjuntos de entrenamiento
     y prueba.

7. guardar_modelo(model, std):
   - Guarda el modelo entrenado y el objeto StandardScaler en el
   directorio 'artifacts'.

Dependencias:
- pandas
- matplotlib
- scikit-learn
- joblib
- yaml
"""

# %%
# Importamos librerias
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import yaml

# Abrimos yaml
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# usamos argparse para inputs y outputs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('input_train') #clean_train
parser.add_argument('output_modelo') #model
parser.add_argument('output_scaler') #std
args = parser.parse_args()

# %%
# Guardamos dataframes desde los datos ya limpios
df = pd.read_csv(f'data/{args.input_train}.csv')
df.head()

# %%
# Separamos variables para entrenamiento y prueba
X = df.drop(['SalePrice'], axis=1)
Y = df['SalePrice']
Train_X, Test_X, Train_Y, Test_Y = \
   train_test_split(X, Y, train_size=config['modeling']['train_size'], test_size=config['modeling']['test_size'], random_state=100)

# %%
# Estandarizamos las variables.

std = StandardScaler()

Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)

Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)

# Mostramos ejemplo para train
Train_X_std.describe()

# %%
# Modelo
# Ejecutamos Regresion Lineal

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

# %%
# Graficamos resultados de predicción

# Entrenamiento
plt.scatter(Train_Y, y_train_pred)
plt.plot([min(Train_Y), max(Train_Y)], [min(Train_Y), max(Train_Y)],
         linestyle='--', color='red', linewidth=2)  # 1:1 line
plt.title('Actual vs Predicted para Entrenamiento')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Test
plt.scatter(Test_Y, y_test_pred)
plt.plot([min(Test_Y), max(Test_Y)], [min(Test_Y), max(Test_Y)],
         linestyle='--', color='red', linewidth=2)  # 1:1 line
plt.title('Actual vs Predicted para Test')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# %%
# Guardamos modelo
joblib.dump(model, f'artifacts/{args.output_modelo}.joblib')
# Guardamos scaler
joblib.dump(std, f'artifacts/{args.output_scaler}.joblib')
