# APD-tarea3
Repositorio sobre la tarea 3 de Arquitectura de Producto de Datos (MGE)

La estructura del repositorio es como sigue:

- notebooks: contiene el notebook ipynb original con el modelo completo.

- data: contiene los archivos csv tanto crudos como preparados para entrenar el modelo, así como el archivo de entrada para realizar la inferencia  y el archivo de predicciones de salida.

- artifacts: contiene el modelo entrenado.

- prep.py: la entrada del script son datos data/raw. La salida del script son datos clean.

- train.py: la entrada del script son datos data/clean. La salida del script es el modelo entrenado model.joblib

- inference.py: la entrada de este script son datos data/inference y el modelo entrenado model.joblib. La salida de este modelo son predicciones en batch que se guardan en data/predictions
