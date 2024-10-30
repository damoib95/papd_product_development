# src/preprocess.py
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

def preprocess(input_file, output_file, features, target):
    # Cargar el dataset
    df = pd.read_csv(input_file)

    # Eliminar filas con valores nulos
    df = df.dropna()

    # Seleccionar solo las columnas necesarias
    columns = features + [target]
    df = df[columns]

    # Normalizar las características
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Guardar el dataset limpio
    df.to_csv(output_file, index=False)
    print(f"Preprocesamiento completado. Datos guardados en {output_file}")

if __name__ == "__main__":
    params_file = "params.yaml"

    # Leer parámetros desde params.yaml
    with open(params_file) as f:
        params = yaml.safe_load(f)

    input_file = params['data']['input_path']
    output_file = params['data']['output_path']
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    preprocess(input_file, output_file, features, target)
