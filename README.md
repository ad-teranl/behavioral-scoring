# Behavioral Scoring MLOps — 2026

Pipeline MLOps completo para deteccion de mora en cartera de consumo.
Basado en el QMD `behavioral_scoring_portafolio_XGB_LGBM.qmd` — Escenario 3.

## Estructura del proyecto

```
2026_behavioral_scoring_prueba/
├── config/
│   └── main.yaml              <- TODA la configuracion (sin hardcodeo)
├── data/
│   └── raw/
│       └── BASE_EJERCICIO_SCORING_31_variables.csv  <- COLOCA AQUI EL CSV
├── models/                    <- pkl generados al entrenar
├── reports/                   <- metricas, plots SHAP, tablas
├── notebooks/                 <- explorar resultados interactivamente
├── src/
│   ├── preprocessing.py       <- limpieza, particion, transformacion, SMOTE
│   ├── feature_engineering.py <- MIUFS, autoencoder, seleccion de variables
│   ├── train_supervised.py    <- XGBoost + LightGBM + GridSearchCV
│   ├── train_unsupervised.py  <- K-Means + DBSCAN + Isolation Forest
│   └── cierre_integrador.py   <- anomalos 2+ modelos -> P(mora) + SHAP
├── train_pipeline.py          <- PUNTO DE ENTRADA (@hydra.main)
├── predict.py                 <- prediccion sobre nuevos datos
├── requirements.txt
└── Makefile
```

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Primera ejecucion: entrena todo desde cero
python train_pipeline.py reentrenar=true

# Ejecuciones siguientes: carga pkl sin reentrenar
python train_pipeline.py

# Cambiar parametros sin tocar el codigo
python train_pipeline.py reentrenar=true xgboost.cv_folds=3
python train_pipeline.py kmeans.n_clusters=4
python train_pipeline.py target.umbral_mora=60

# Generar predicciones
python predict.py
```

## Flujo del pipeline

```
CSV (31 variables, sep=";")
    -> Limpieza (duplicados, fechas, DIAS_MADURACION, DIAS_RESTANTES)
    -> TARGET: DIASMORA > 30 dias
    -> Particion por operacion (StratifiedGroupKFold en NUM_OPER)
    -> Transformaciones (log1p + StandardScaler / MinMaxScaler)
    -> Encoding categorico (ordinal + binario + OHE)
    -> SMOTE (solo entrenamiento supervisado)
    -> XGBoost + LightGBM (11 variables exogenas)
    -> K-Means + DBSCAN + Isolation Forest (18 variables)
    -> Cierre integrador: anomalos en 2+ modelos -> P(mora) + SHAP
```

## Modelos

| Tipo | Modelo | Variables | Metrica objetivo |
|------|--------|-----------|-----------------|
| Supervisado | XGBoost | 11 exogenas | F1 (GridSearchCV) |
| Supervisado | LightGBM | 11 exogenas | F1 (GridSearchCV) |
| No supervisado | K-Means | 18 MIUFS+AE | Silhouette |
| No supervisado | DBSCAN | 18 MIUFS+AE | Pureza vs TARGET |
| No supervisado | Isolation Forest | 18 MIUFS+AE | Precision vs TARGET |

## Salidas en reports/

- `metricas_supervisados.csv` — F1, AUC, KS, Brier, MCC
- `resumen_nosupervisados.csv` — anomalos detectados y precision vs mora real
- `anomalos_con_probabilidad.csv` — tabla de operaciones anomalas con P(mora)
- `shap_beeswarm_anomalos.png` — importancia global en anomalos
- `shap_waterfall_top{1-5}.png` — waterfall por operacion de mayor riesgo
- `cm_*.png` — matrices de confusion
- `roc_*.png` — curvas ROC

## Dashboard de monitoreo (Streamlit)

Flujo de uso (siempre en consola desde la raiz, o con los .bat):

```bash
python train_pipeline.py     # 1. entrena o carga los modelos (models/)
python score_cartera.py      # 2. puntua la cartera vigente (reports/scoring/)
python -m streamlit run app/Inicio.py   # 3. abre el tablero
```

En Windows: doble clic en `actualizar_scoring.bat` y luego en `iniciar_dashboard.bat`.

| Pagina | Proposito |
|--------|-----------|
| Inicio | Anomalias por consenso de K-Means, DBSCAN e Isolation Forest + confirmacion supervisada |
| Alerta preventiva | Socios al dia que se acercan al umbral (distancia y meses al umbral) |
| Sobre umbral | Al dia con P >= umbral (intervencion) y ya en mora (cobranza) |
| Consulta operacion | P(mora), semaforo, trayectoria mes a mes y SHAP waterfall |

Cartera vigente = operaciones presentes en el ultimo corte (31/12/2019).
El dashboard solo lee los CSV de `reports/scoring/`, no necesita los modelos para funcionar.

## Cortes mensuales nuevos (sin reentrenar)

Los escaladores y codificadores del entrenamiento 2017-2019 se congelan en
`models/artefactos_scoring.pkl` (clase `TransformadorCartera`). Los cortes nuevos
solo se transforman con ellos; nunca se vuelve a ajustar nada.

```bash
python congelar_transformador.py                 # una sola vez
python actualizar_corte.py ruta/corte.csv        # cada mes (o desde el tablero)
```

- El CSV original nunca se modifica; los cortes se guardan en `data/raw/cortes_nuevos/`.
- Ejemplo simulado: `data/ejemplos/corte_2020_01_SIMULADO.csv` (585 continuan + 5 nuevas).
- Pagina **Cargar nuevo corte**: valida, muestra vista previa por modelo e incorpora con un boton.
- Pagina **Anomalias por modelo**: lo que senala cada detector por separado.
