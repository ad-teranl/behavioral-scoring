"""
predict.py
==========
Carga los modelos entrenados y genera P(mora) para nuevas operaciones.

Uso:
  python predict.py                          <- usa main.yaml
  python predict.py data.path=data/raw/nuevo.csv

El CSV nuevo debe tener el mismo formato que el CSV de entrenamiento
(mismas columnas, mismo separador). El pipeline aplica exactamente
las mismas transformaciones usadas en train_pipeline.py gracias a
que los scalers y encoders se cargan desde models/.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import hydra
import joblib
import numpy as np
import pandas as pd
from hydra import utils
from omegaconf import DictConfig

import src.preprocessing as preprocessing


@hydra.main(config_path="config", config_name="main", version_base=None)
def run_predict(config: DictConfig):
    """
    Aplica el pipeline de preprocesamiento y genera P(mora) con XGBoost.
    """
    proyecto_root = utils.get_original_cwd() + "/"
    config.data.path     = proyecto_root + config.data.path
    config.paths.models  = proyecto_root + config.paths.models
    config.paths.reports = proyecto_root + config.paths.reports
    ruta_models  = config.paths.models
    ruta_reports = config.paths.reports

    print("\n[Predict] Cargando modelos...")
    modelo_xgb  = joblib.load(f"{ruta_models}modelo_XGBoost.pkl")
    modelo_lgbm = joblib.load(f"{ruta_models}modelo_LightGBM.pkl")
    print("  XGBoost y LightGBM cargados correctamente.")

    # Preprocesar los datos nuevos con el mismo pipeline
    print("\n[Predict] Preprocesando datos nuevos...")
    datasets = preprocessing.run_preprocessing(config)

    features_sup = list(config.features_supervisado)
    X_te = datasets['X_test_sup'][features_sup]
    y_te = datasets['y_test']

    # Predicciones
    proba_xgb  = modelo_xgb.predict_proba(X_te)[:, 1]
    proba_lgbm = modelo_lgbm.predict_proba(X_te)[:, 1]
    proba_ens  = (proba_xgb + proba_lgbm) / 2  # ensemble simple

    df_result = pd.DataFrame({
        'NUM_OPER':     X_te.index,
        'P_mora_XGB':  proba_xgb.round(4),
        'P_mora_LGBM': proba_lgbm.round(4),
        'P_mora_ENS':  proba_ens.round(4),
        'TARGET_real': y_te.values,
        'Pred_XGB':    (proba_xgb >= 0.45).astype(int),
        'Zona_gris':   ((proba_ens >= 0.40) & (proba_ens <= 0.60)).astype(int),
    })

    print(f"\n[Predict] Resultados sobre {len(df_result)} operaciones de prueba:")
    print(f"  P(mora) promedio XGBoost:  {proba_xgb.mean():.4f}")
    print(f"  P(mora) promedio LightGBM: {proba_lgbm.mean():.4f}")
    print(f"  P(mora) promedio Ensemble: {proba_ens.mean():.4f}")
    print(f"  En zona gris (0.40-0.60):  {df_result['Zona_gris'].sum()}")

    # Top 10 operaciones de mayor riesgo
    top10 = df_result.nlargest(10, 'P_mora_ENS')[
        ['NUM_OPER', 'P_mora_XGB', 'P_mora_LGBM', 'P_mora_ENS', 'TARGET_real']
    ]
    print(f"\n  Top 10 operaciones de mayor riesgo (ensemble):")
    print(top10.to_string(index=False))

    # Exportar
    ruta_salida = f"{ruta_reports}predicciones.csv"
    df_result.to_csv(ruta_salida, index=False)
    print(f"\n  Predicciones exportadas: {ruta_salida}")


if __name__ == "__main__":
    run_predict()
