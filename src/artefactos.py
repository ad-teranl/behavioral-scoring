"""
artefactos.py
=============
Guarda en models/artefactos_scoring.pkl todo lo que el scoring necesita
para NO volver a partir ni reajustar nada con datos nuevos:

  - transformador         : TransformadorCartera (escaladores/codificadores de train)
  - particion             : NUM_OPER -> train / val / test (operaciones nuevas = 'nuevo')
  - cluster_riesgo_kmeans : cluster de mayor mora, identificado en train
  - corte_entrenamiento   : ultimo corte visto en el entrenamiento
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.transformador import TransformadorCartera

NOMBRE = "artefactos_scoring.pkl"


def construir(datasets, config):
    transformador = TransformadorCartera(
        datasets['scalers_sup'], datasets['scalers_nosup'],
        datasets['ohe_sup'], datasets['ohe_nosup'])

    particion = {}
    for part in ['train', 'val', 'test']:
        for op in datasets[f'X_{part}_sup'].index.unique():
            particion[int(op)] = part

    kmeans = joblib.load(os.path.join(config.paths.models, "modelo_KMeans.pkl"))
    Xtr = datasets['X_train_nosup'][list(config.features_nosupervisado)]
    tasa = pd.Series(datasets['y_train'].values).groupby(kmeans.predict(Xtr)).mean()

    fechas = pd.concat([datasets['fecha_train'], datasets['fecha_val'], datasets['fecha_test']])
    return {
        'transformador': transformador,
        'particion': particion,
        'cluster_riesgo_kmeans': int(tasa.idxmax()),
        'corte_entrenamiento': pd.Timestamp(fechas.max()),
    }


def guardar(artefactos, config):
    ruta = os.path.join(config.paths.models, NOMBRE)
    joblib.dump(artefactos, ruta)
    print(f"[Artefactos] Guardados en {ruta}")
    print(f"   Operaciones con particion: {len(artefactos['particion'])} | "
          f"cluster riesgo K-Means: {artefactos['cluster_riesgo_kmeans']} | "
          f"corte entrenamiento: {artefactos['corte_entrenamiento']:%d/%m/%Y}")


def cargar(config):
    ruta = os.path.join(config.paths.models, NOMBRE)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta una vez: python congelar_transformador.py")
    return joblib.load(ruta)
