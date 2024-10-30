# src/train.py
import pandas as pd
import joblib
import sys
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


def train(input_file, model_path, params_file):
    # Cargar el dataset limpio
    df = pd.read_csv(input_file)

    # Leer los hiperparámetros
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # Separar características y variable objetivo
    features = params['preprocessing']['features']
    target = params['preprocessing']['target']

    X = df[features]
    y = df[target]

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params['train']['test_size'], random_state=params['train']['random_state']
    )

    # Definición de modelos
    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'hyperparameters': params['train']['linear_regression']['hyperparameters']
        },
        'RandomForest': {
            'model': RandomForestRegressor(),
            'hyperparameters': params['train']['random_forest']['hyperparameters']
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(),
            'hyperparameters': params['train']['gradient_boosting']['hyperparameters']
        }
    }

    best_model = None
    best_score = float('inf')
    best_model_name = ""
    all_results = []

    # Entrenar el modelo
    for model_name, model_config in models.items():
        model = model_config['model']
        hyperparameters = model_config['hyperparameters']

        # Implementar GridSearchCV para búsqueda de hiperparámetros
        search = GridSearchCV(model, hyperparameters, cv=5, scoring='neg_mean_squared_error')
        search.fit(X_train, y_train)

        # Almacenar resultados
        all_results.append({
            'model': model_name,
            'best_score': -search.best_score_,  # Convertir a positivo ya que GridSearchCV devuelve negativo
            'best_params': search.best_params_
        })

        # Actualizar el mejor modelo si es necesario
        if -search.best_score_ < best_score:
            best_score = -search.best_score_
            best_model = search.best_estimator_
            best_model_name = model_name

        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Guardar el modelo entrenado
        model_file = f"{model_path}/{model_name}_model.joblib"
        joblib.dump(search.best_estimator_, model_file)
        print(f"Modelo {model_name} entrenado y guardado en {model_file}")

    # Mostrar resultados
    for result in all_results:
        print(f"Modelo: {result['model']}, Mejor Score (MSE): {result['best_score']}, Mejores Parámetros: {result['best_params']}")

    print(f"\nMejor modelo: {best_model_name} con un score (MSE) de {best_score}")

    # Guardar el mejor modelo como champion_model.joblib
    champion_model_file = f"{model_path}/champion_model.joblib"
    joblib.dump(best_model, champion_model_file)
    print(f"Mejor modelo guardado como {champion_model_file}")


if __name__ == "__main__":
    params_file = "params.yaml"

    with open(params_file) as f:
        params = yaml.safe_load(f)

    input_file = params['data']['output_path']
    model_path = params['train']['model_path']

    train(input_file, model_path, params_file)
