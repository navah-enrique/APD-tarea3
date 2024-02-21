"""
Script: inference.py

Este script se encarga de realizar inferencias utilizando un modelo de
regresión lineal previamente entrenado. Carga el modelo y el objeto
StandardScaler entrenados con 'train.py', luego aplica el modelo a datos
de inferencia, guardando las predicciones en un archivo CSV.

Uso:
1. Se requiere haber ejecutado 'train.py' previamente para entrenar el
modelo y guardar los artefactos.
2. Ejecutar este script para realizar inferencias y guardar las
predicciones en 'data/predictions.csv'.

Funciones:
1. cargar_modelo_y_escalador():
   - Carga el modelo entrenado ('model.joblib') y el objeto StandardScaler
     ('std.joblib') desde el directorio 'artifacts'.

2. realizar_inferencias(modelo, escalador):
   - Carga datos de inferencia limpios ('clean_inference.csv') y realiza
   inferencias utilizando el modelo y el escalador.
   - Guarda las predicciones en un archivo CSV llamado 'predictions.csv'
     en el directorio 'data'.

Dependencias:
- pandas
- joblib
"""
# %%
# Importamos librerias
import pandas as pd
import joblib

# %%
loaded_model = joblib.load('artifacts/model.joblib')
loaded_scaler = joblib.load('artifacts/std.joblib')
inference = pd.read_csv('data/clean_inference.csv')
ids = pd.read_csv('data/raw_inference.csv')

inf_std = loaded_scaler.transform(inference)
inf_std = pd.DataFrame(inf_std, columns=inference.columns)
pred_y = loaded_model.predict(inf_std)

result = pd.DataFrame({'Id': ids['Id'], 'SalePrice': pred_y})
result.to_csv('data/predictions.csv', index=False)
