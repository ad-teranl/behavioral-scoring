"""
train_unsupervised.py
=====================
Entrena tres detectores de anomalias sobre las 18 variables no supervisadas:
  - K-Means
  - DBSCAN
  - Isolation Forest

Logica reentrenar (controlada desde main.yaml):
  reentrenar: true  -> entrena desde cero y guarda pkl
  reentrenar: false -> carga pkl existentes

Validacion externa: aunque no hay label en el entrenamiento,
se usa el TARGET real para calcular pureza y precision
de los clusters y anomalias detectados.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score


def _guardar_tabla_anomalias(nombre, indices_anomalos, y_test,
                              ruta_reports):
    """
    Calcula cuantos anomalos detectados tienen TARGET=1 (mora real).
    """
    if len(indices_anomalos) == 0:
        print(f"  [{nombre}] Sin anomalos detectados.")
        return pd.DataFrame()

    y_anomalos = y_test.iloc[indices_anomalos] if hasattr(y_test, 'iloc') \
                 else y_test[indices_anomalos]

    total    = len(indices_anomalos)
    en_mora  = int(y_anomalos.sum())
    precision = en_mora / total if total > 0 else 0

    print(f"  [{nombre}] Anomalos detectados: {total} | Con mora real: {en_mora} | Precision: {precision:.4f}")

    resumen = pd.DataFrame([{
        'Modelo':    nombre,
        'Anomalos':  total,
        'Con_mora':  en_mora,
        'Precision': round(precision, 4)
    }])
    return resumen


def train_kmeans(datos_nosup, config, ruta_models, ruta_reports, reentrenar):
    """
    K-Means: identifica operaciones en el cluster de mayor mora como anomalas.
    """
    X_tr = datos_nosup['X_tr_nosup']
    X_te = datos_nosup['X_te_nosup']
    y_te = datos_nosup['y_te']

    ruta_model = f"{ruta_models}modelo_KMeans.pkl"
    cfg = config.kmeans

    if reentrenar or not os.path.exists(ruta_model):
        print("\n[K-Means] Entrenando...")
        modelo = KMeans(
            n_clusters=cfg.n_clusters,
            random_state=cfg.random_state,
            init=cfg.init,
            n_init=cfg.n_init,
            max_iter=cfg.max_iter
        )
        modelo.fit(X_tr)
        joblib.dump(modelo, ruta_model)
        sil = silhouette_score(X_tr, modelo.labels_)
        print(f"  [K-Means] Silhouette score (train): {sil:.4f}")
        print(f"  [K-Means] Guardado en: {ruta_model}")
    else:
        print("\n[K-Means] Cargando modelo existente (reentrenar=false)")

    modelo = joblib.load(ruta_model)

    # Identificar cluster de mayor mora como anomalo
    clusters_te = modelo.predict(X_te)
    df_clusters = pd.DataFrame({'cluster': clusters_te, 'TARGET': y_te.values})
    mora_por_cluster = df_clusters.groupby('cluster')['TARGET'].mean()
    cluster_riesgo   = mora_por_cluster.idxmax()
    print(f"  [K-Means] Cluster de mayor riesgo: {cluster_riesgo} (mora={mora_por_cluster[cluster_riesgo]:.4f})")

    idx_anomalos = np.where(clusters_te == cluster_riesgo)[0]
    resumen = _guardar_tabla_anomalias("K-Means", idx_anomalos, y_te, ruta_reports)

    return modelo, idx_anomalos, resumen


def train_dbscan(datos_nosup, config, ruta_models, ruta_reports, reentrenar):
    """
    DBSCAN: el ruido (-1) se considera anomalo.
    """
    X_tr = datos_nosup['X_tr_nosup']
    X_te = datos_nosup['X_te_nosup']
    y_te = datos_nosup['y_te']

    ruta_model = f"{ruta_models}modelo_DBSCAN.pkl"
    cfg = config.dbscan

    if reentrenar or not os.path.exists(ruta_model):
        print("\n[DBSCAN] Entrenando...")
        modelo = DBSCAN(
            eps=cfg.eps,
            min_samples=cfg.min_samples,
            metric=cfg.metric
        )
        modelo.fit(X_tr)
        n_clusters = len(set(modelo.labels_)) - (1 if -1 in modelo.labels_ else 0)
        n_ruido    = list(modelo.labels_).count(-1)
        print(f"  [DBSCAN] Clusters: {n_clusters} | Ruido: {n_ruido}")
        joblib.dump(modelo, ruta_model)
        print(f"  [DBSCAN] Guardado en: {ruta_model}")
    else:
        print("\n[DBSCAN] Cargando modelo existente (reentrenar=false)")

    modelo = joblib.load(ruta_model)

    # En test: puntos no asignados a ningun cluster = anomalos
    labels_te    = modelo.fit_predict(X_te)  # DBSCAN no tiene predict nativo
    idx_anomalos = np.where(labels_te == -1)[0]
    resumen = _guardar_tabla_anomalias("DBSCAN", idx_anomalos, y_te, ruta_reports)

    return modelo, idx_anomalos, resumen


def train_isolation_forest(datos_nosup, config, ruta_models, ruta_reports, reentrenar):
    """
    Isolation Forest: score < 0 indica anomalia.
    """
    X_tr = datos_nosup['X_tr_nosup']
    X_te = datos_nosup['X_te_nosup']
    y_te = datos_nosup['y_te']

    ruta_model = f"{ruta_models}modelo_IsolationForest.pkl"
    cfg = config.isolation_forest

    max_samples = cfg.max_samples if cfg.max_samples != 'auto' else 'auto'

    if reentrenar or not os.path.exists(ruta_model):
        print("\n[Isolation Forest] Entrenando...")
        modelo = IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
            max_samples=max_samples
        )
        modelo.fit(X_tr)
        joblib.dump(modelo, ruta_model)
        print(f"  [Isolation Forest] Guardado en: {ruta_model}")
    else:
        print("\n[Isolation Forest] Cargando modelo existente (reentrenar=false)")

    modelo = joblib.load(ruta_model)

    preds_te     = modelo.predict(X_te)   # -1 = anomalia, 1 = normal
    idx_anomalos = np.where(preds_te == -1)[0]
    resumen = _guardar_tabla_anomalias("Isolation Forest", idx_anomalos, y_te, ruta_reports)

    return modelo, idx_anomalos, resumen


def run(datos_nosup, config, reentrenar):
    """
    Punto de entrada para entrenamiento no supervisado.
    Retorna modelos, indices anomalos y resumen de validacion.
    """
    ruta_models  = config.paths.models
    ruta_reports = config.paths.reports

    print("\n" + "="*55)
    print("MODELOS NO SUPERVISADOS")
    print("="*55)

    modelo_km,  idx_km,  res_km  = train_kmeans(
        datos_nosup, config, ruta_models, ruta_reports, reentrenar
    )
    modelo_db,  idx_db,  res_db  = train_dbscan(
        datos_nosup, config, ruta_models, ruta_reports, reentrenar
    )
    modelo_if,  idx_if,  res_if  = train_isolation_forest(
        datos_nosup, config, ruta_models, ruta_reports, reentrenar
    )

    # Tabla resumen
    resumenes = [r for r in [res_km, res_db, res_if] if not r.empty]
    if resumenes:
        tabla = pd.concat(resumenes, ignore_index=True)
        print(f"\n{'='*55}")
        print("RESUMEN NO SUPERVISADOS")
        print(tabla.to_string(index=False))
        tabla.to_csv(f"{ruta_reports}resumen_nosupervisados.csv", index=False)

    return {
        'kmeans':          modelo_km,
        'dbscan':          modelo_db,
        'isolation_forest':modelo_if,
        'idx_km':          idx_km,
        'idx_db':          idx_db,
        'idx_if':          idx_if,
    }
