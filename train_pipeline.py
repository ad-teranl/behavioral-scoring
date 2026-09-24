"""
train_pipeline.py
=================
Punto de entrada unico del proyecto.

Uso:
  python train_pipeline.py                              <- usa main.yaml
  python train_pipeline.py reentrenar_lightgbm=true    <- solo reentrenar LightGBM
  python train_pipeline.py reentrenar_xgboost=true     <- solo reentrenar XGBoost
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import hydra
from hydra import utils
from omegaconf import DictConfig

import src.preprocessing       as preprocessing
import src.feature_engineering as feature_engineering
import src.train_supervised    as train_supervised
import src.train_unsupervised  as train_unsupervised
import src.cierre_integrador   as cierre_integrador
import src.artefactos          as artefactos


@hydra.main(config_path="config", config_name="main", version_base=None)
def run_pipeline(config: DictConfig):

    proyecto_root = utils.get_original_cwd() + "/"
    config.data.path     = proyecto_root + config.data.path
    config.paths.models  = proyecto_root + config.paths.models
    config.paths.reports = proyecto_root + config.paths.reports

    reentrenar_xgb  = config.reentrenar_xgboost
    reentrenar_lgbm = config.reentrenar_lightgbm

    print("\n" + "="*60)
    print("BEHAVIORAL SCORING MLOPS — PIPELINE COMPLETO")
    print(f"  reentrenar_xgboost  = {reentrenar_xgb}")
    print(f"  reentrenar_lightgbm = {reentrenar_lgbm}")
    print("="*60)

    # PASO 1: PREPROCESAMIENTO
    datasets = preprocessing.run_preprocessing(config)

    # PASO 2: SELECCION DE VARIABLES
    datos_sup   = feature_engineering.select_features_supervisado(datasets, config)
    datos_nosup = feature_engineering.select_features_nosupervisado(datasets, config)

    # PASO 3: MODELOS SUPERVISADOS
    modelos_sup = train_supervised.run(
        datos_sup, config, reentrenar_xgb, reentrenar_lgbm
    )

    # PASO 4: MODELOS NO SUPERVISADOS
    modelos_nosup = train_unsupervised.run(datos_nosup, config, False)

    # PASO 5: CIERRE INTEGRADOR
    cierre_integrador.run(
        datos_sup=datos_sup,
        datos_nosup=datos_nosup,
        modelos_supervisados=modelos_sup,
        modelos_nosupervisados=modelos_nosup,
        config=config
    )

    # PASO 6: CONGELAR TRANSFORMADOR (para scoring de cortes nuevos)
    artefactos.guardar(artefactos.construir(datasets, config), config)

    print("\n" + "="*60)
    print("PIPELINE COMPLETADO")
    print(f"  Modelos en:  {config.paths.models}")
    print(f"  Reportes en: {config.paths.reports}")
    print("="*60)


if __name__ == "__main__":
    run_pipeline()
