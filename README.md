# APD-tarea3
Repositorio sobre la tarea 3 de Arquitectura de Producto de Datos (MGE)

La estructura del repositorio es como sigue:

- Notebooks: contiene el notebook ipynb original con el modelo completo.

- Data: contiene los csvs tanto crudos como preparados para entrenar el modelo.

- Artifacts: contiene el modelo entrenado.

- prep.py: la entrada del script son datos data/raw. La salida del script son datos prep

- train.py: la entrada del script son datos data/prep. La salida del script es el modelo entrenado model.joblib

- inference.py: la entrada de este script son datos data/inference y el modelo entrenado model.joblib. La salida de este modelo son predicciones en batch que se guardan en data/predictions
