# Advanced Classification Pipeline & XGBoost Optimization Framework

Este repositorio contiene un ecosistema de **Aprendizaje Automático** de alto rendimiento desarrollado en **Python**, centrado en el uso de **XGBoost (Extreme Gradient Boosting)** para resolver problemas complejos de clasificación estructurada. El proyecto destaca por la automatización de la ingeniería de características mediante pipelines y un robusto esquema de validación estadística diseñado para maximizar la capacidad predictiva y la eficiencia computacional.

## 🚀 Capacidades Técnicas e Ingeniería de Características

* **Modelado con XGBoost de Vanguardia:** Implementación del algoritmo `XGBClassifier`, aprovechando el poder del aprendizaje por transferencia de gradiente, regularización interna ($L_1$ y $L_2$) y manejo optimizado de datos dispersos.
* **Pipelines de Transformación Segura:** Integración de `ColumnTransformer` para automatizar flujos aislados de preprocesamiento: imputación de valores faltantes (`SimpleImputer`), escalamiento estadístico (`StandardScaler`), codificación nominal (`OneHotEncoder`), codificación ordinal (`OrdinalEncoder`) y codificación de etiquetas objetivo (`LabelEncoder`).
* **Estrategia de Validación Ultra-Robusta:** Implementación de técnicas de remuestreo avanzadas como **`RepeatedStratifiedKFold`** y **`RepeatedKFold`**, mitigando el desbalance de clases y reduciendo drásticamente la varianza en la evaluación del modelo.
* **Evaluación Dual y Multidimensión:** Estructuración de un entorno métrico capaz de evaluar rendimiento tanto en escenarios de clasificación (*Accuracy*, *Precision*, *Recall*, $F_1$-Score, *ROC-AUC* vía `RocCurveDisplay`) como en análisis de desviaciones continuas (`MSE`, `MAE`, $R^2$ Score).

## 🛠️ Stack Tecnológico

* **Lenguaje Core:** Python 3.x
* **Estructuras de Datos:** `Pandas`, `NumPy`
* **Visualización Estadística:** `Seaborn`, `Matplotlib`
* **Modelado y Preprocesamiento:** `Scikit-Learn` (Sklearn)
* **Algoritmo Core:** `XGBoost` (eXtreme Gradient Boosting)

## ⚙️ Arquitectura del Pipeline y Mitigación de Errores

El diseño de este framework se enfocó en resolver problemas críticos de la puesta en producción de modelos predictivos:

1. **Inmunidad al Data Leakage:** Al empaquetar de forma hermética el procesamiento (`SimpleImputer`, `StandardScaler`) junto con el estimador dentro de un `Pipeline`, los parámetros de transformación se calculan exclusivamente con el set de entrenamiento en cada pliegue de la validación cruzada, impidiendo la contaminación de datos de prueba.
2. **Optimización Eficiente del Espacio de Parámetros:** Uso coordinado de búsquedas exhaustivas (`GridSearchCV`) y estocásticas (`RandomizedSearchCV`) para sintonizar los hiperparámetros críticos de XGBoost (como la tasa de aprendizaje `learning_rate`, profundidad máxima `max_depth`, y penalizaciones de regularización).
3. **Análisis de Capacidad Discriminatoria:** Evaluación visual del compromiso entre la tasa de verdaderos positivos y falsos positivos mediante curvas ROC, garantizando que el modelo sea capaz de separar de manera óptima las clases de negocio estudiadas.

## 📌 Flujo de Datos del Sistema

$$\text{Datos Estructurados} \longrightarrow \text{Pipeline (Imputación + Encoders + Escalamiento)} \longrightarrow \text{Repeated Stratified K-Fold} \longrightarrow \text{Ajuste Fino (Grid/Random Search)} \longrightarrow \text{Evaluación (F1 / ROC-AUC / Regresión)}$$

## 🔧 Configuración e Instalación Local

Para replicar los experimentos, auditar las matrices de confusión o ejecutar el modelo en tu máquina local, sigue estos pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/longaresf/xgboost-advanced-classification-pipeline.git](https://github.com/longaresf/xgboost-advanced-classification-pipeline.git)
   ```
2. Ingresar al directorio de trabajo:
   Bash
   cd xgboost-advanced-classification-pipeline

3. Configurar el entorno virtual (Recomendado):
   Bash
   python -m venv venv
   source venv/bin/activate  # En Windows usa: venv\Scripts\activate

4. Instalar dependencias e intérpretes:
   Bash
   pip install -r requirements.txt

5. Ejecutar el pipeline:
   Inicia tu entorno de Jupyter Notebook o ejecuta el script principal para visualizar las curvas de aprendizaje y rendimiento de XGBoost:

   Bash
   python main.py  # Reemplaza por el nombre de tu archivo principal si difiere

✒️ Autor

    Francisco Longares - Científico de Datos & Desarrollador de Soluciones IA - longaresf
