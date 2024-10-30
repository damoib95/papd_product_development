# src/evaluate.py
import pandas as pd
import joblib
import json
import yaml
from sklearn.metrics import mean_squared_error, r2_score, classification_report

def evaluate(input_file, model_file, metrics_file, feature_importance_file, features, target):
    # Cargar el dataset limpio
    df = pd.read_csv(input_file)

    # Leer los parámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Separar características y variable objetivo
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X = df[features]
    y = df[target]

    # Cargar el modelo entrenado
    model = joblib.load(model_file)

    # Realizar predicciones
    predictions = model.predict(X)

    # Calcular métricas
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    # Guardar métricas en un archivo JSON
    metrics = {
        'mse': mse,
        'r2': r2
    }

    # Guardar métricas en un archivo JSON
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas guardadas en {metrics_file}")

    # Obtener importancia de características si el modelo lo permite
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Guardar la importancia de características en un CSV
        importance_df.to_csv(feature_importance_file, index=False)
        print(f"Importancia de características guardada en {feature_importance_file}")
    else:
        print("El modelo no tiene importancia de características disponible.")

if __name__ == "__main__":
    params_file = 'params.yaml'
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Argumentos: archivo de entrada, archivo del modelo, archivo de métricas, archivo de parámetros
    input_file=params['data']['output_path']
    model_file=params['evaluate']['model_file']
    metrics_file=params['evaluate']['metrics_file']
    feature_importance_file=params['evaluate']['feature_importance_file']
    features=params['preprocessing']['features']
    target=params['preprocessing']['target']

    evaluate(input_file, model_file, metrics_file, feature_importance_file, features, target)
