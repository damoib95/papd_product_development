# papd_pd_lab_1
## Laboratorio 1: Gestión de Experimentación y Modelos con DVC Automated Machine Learning (AutoML)

Código y materiales utilizados para el Laboratorio 1 del curso **Product Development** del postgrado en **Análisis y Predicción de Datos** de la **Maestría en Data Science** en la **Universidad Galileo**.

### Configuración del Proyecto

1. **Inicializa Git y DVC**
   ```bash
   git init
   dvc init
   dvc config core.autostage true
   ```

2. **Crea el Pipeline con DVC**

   - **Preprocesamiento**: Procesa los datos de entrada y genera `preprocessed_data.csv`.
     ```bash
     dvc stage add -n preprocess -d src/preprocess.py -d data/data.csv -o data/preprocessed_data.csv python src/preprocess.py
     ```

   - **Entrenamiento**: Entrena el modelo y guarda el archivo `model.joblib`.
     ```bash
     dvc stage add -n train -d src/train.py -d data/preprocessed_data.csv -o models/model.joblib python src/train.py
     ```

   - **Evaluación**: Evalúa el modelo y guarda las métricas en `metrics.json`.
     ```bash
     dvc stage add -n evaluate -p evaluate -d src/evaluate.py -d data/preprocessed_data.csv -d models/champion_model.joblib -o metrics.json python src/evaluate.py
     ```

3. **Ejecutar el Pipeline Completo**

   Una vez configuradas todas las etapas, puedes ejecutar el pipeline completo usando:
   ```bash
   dvc repro
   ```

### Ejecución del Proyecto

1. Clona este repositorio y navega al directorio del proyecto:
   ```bash
   git clone https://github.com/damoib95/papd_product_development/tree/main
   cd papd_product_development
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta el pipeline completo de DVC para procesar datos, entrenar y evaluar el modelo:
   ```bash
   dvc repro
   ```
