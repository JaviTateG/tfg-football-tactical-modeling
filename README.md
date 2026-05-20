# Modelado de Estrategias de Juego mediante Teoría de Grafos y Aprendizaje Automático

Trabajo de Fin de Grado - Grado en Ingeniería Matemática  
Universidad Francisco de Vitoria, Madrid | Convocatoria Junio 2026  
Autor: Javier Gómez Jiménez

## Descripción

Este proyecto analiza el comportamiento táctico colectivo del FC Barcelona en La Liga 
(temporadas 2018/19, 2019/20 y 2020/21) mediante grafos de pases y modelos de 
aprendizaje automático. A partir de datos de eventos de partido de StatsBomb Open Data 
y estadísticas históricas de Understat, se desarrollaron cuatro módulos:

- **Análisis de grafos de pases** — métricas de centralidad, análisis dinámico temporal 
  y simulación táctica
- **Clasificador de estilos tácticos** — Random Forest sobre métricas del grafo 
  (accuracy 86,3%)
- **Anticipación de jugadas peligrosas** — modelos LSTM y GRU sobre secuencias 
  temporales de eventos (AUC-ROC 0,7226)
- **Predicción pre-partido** — sistema que combina métricas del grafo con historial 
  de expected goals de Understat (accuracy 57,9%)

## Requisitos

Python 3.10. Instalar dependencias:

```bash
pip install -r requirements.txt
```

Principales librerías: `statsbombpy`, `networkx`, `tensorflow==2.21`, `scikit-learn==1.3`, 
`pandas`, `numpy`, `matplotlib`, `seaborn`, `ipywidgets`, `understat`, `aiohttp`, `joblib`.

---

## Cómo ejecutar

### Paso previo obligatorio — descarga de datos de Understat

Antes de ejecutar el notebook 07 por primera vez, descargar los datos de Understat 
desde terminal. El cliente asíncrono de Understat es incompatible con el event loop 
de Jupyter, por lo que la descarga debe realizarse de forma independiente:

```bash
cd tfg-football-tactical-modeling
python src/data/download_understat.py
```

Esto descarga los datos de 7 temporadas de La Liga para todos los equipos y genera 
los archivos `understat_team_features.csv` y `understat_matches.csv` en `data/processed/`.

### Orden de ejecución de los notebooks

Los notebooks deben ejecutarse en orden, ya que cada uno consume outputs del anterior:

| Notebook | Descripción | Outputs generados |
|---|---|---|
| `01_data_exploration` | Grafos y análisis táctico del partido de referencia | `metrics_*.csv`, `barcelona_style_data.csv` |
| `02_feature_engineering` | Dataset completo de 102 partidos | `tactical_features_full.csv` |
| `03_barcelona_tactical_classification` | Clasificador de estilos tácticos | `barcelona_style_metrics.csv`, modelos `.pkl` |
| `05_dangerous_plays_prediction` | Modelos LSTM y GRU | Modelos serializados |
| `07_match_prediction` | Predicción pre-partido con selector interactivo | `barca_result_predictor.pkl` |

Los datos de StatsBomb se descargan automáticamente mediante la API de `statsbombpy` 
al ejecutar los notebooks. No es necesario descargarlos manualmente.

---

## Datos

**StatsBomb Open Data**  
Disponible en: https://github.com/statsbomb/open-data  
Licencia abierta para uso académico y no comercial. Se usa la API oficial de Python 
(`statsbombpy`). El proyecto utiliza los partidos de FC Barcelona en La Liga, 
temporadas 2018/19 (season_id=4), 2019/20 (season_id=42) y 2020/21 (season_id=90), 
bajo competition_id=11.

**Understat**  
Disponible en: https://understat.com  
Estadísticas de expected goals para todas las temporadas de La Liga desde 2014/15. 
Acceso gratuito para usos no comerciales mediante la librería `understat`.

---

## Resultados principales

| Módulo | Métrica | Resultado |
|---|---|---|
| Clasificador de estilos | Accuracy test | 0,8627 |
| Clasificador de estilos | AUC-ROC | 0,9704 |
| Clasificador de estilos | CV (5-fold) | 0,8185 ± 0,0132 |
| LSTM jugadas peligrosas | F1-Score | 0,7323 |
| LSTM jugadas peligrosas | AUC-ROC | 0,7226 |
| Predicción pre-partido | Accuracy test | 0,5789 |
| Predicción pre-partido | MAE xG local | 0,8481 |
| Predicción pre-partido | MAE xG visitante | 0,6876 |

**Hallazgo destacado:** Barcelona obtiene su mayor win rate con estilo equilibrado 
(0,750) frente a posesión alta (0,690), paradoja táctica que emerge directamente 
de los datos sin ninguna hipótesis previa.

---

## Contexto académico

Este repositorio corresponde al Trabajo de Fin de Grado del Grado en Ingeniería 
Matemática de la Universidad Francisco de Vitoria (Madrid), convocatoria de junio 
de 2026. El código, los datos y los resultados son completamente reproducibles a 
partir de fuentes abiertas y gratuitas.
