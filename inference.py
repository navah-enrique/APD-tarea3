# %%
#Importamos librerias
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


