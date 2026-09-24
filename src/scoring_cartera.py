"""
scoring_cartera.py
==================
Puntua la cartera usando artefactos CONGELADOS (sin repartir ni reajustar):

  historico original + data/raw/cortes_nuevos/*.csv
      -> preparar_crudo -> TransformadorCartera.transform()  (solo transform)
      -> modelos pkl -> P(mora) + anomalias por modelo
      -> cartera vigente del ultimo corte + trayectorias + SHAP
      -> reports/scoring/*.csv  (lo unico que lee el dashboard)

Tambien ofrece funciones para el dashboard:
  validar_corte()        -> revisa un CSV nuevo antes de incorporarlo
  puntuar_corte_nuevo()  -> vista previa de un corte sin guardarlo
  incorporar_corte()     -> guarda el corte y regenera todo el scoring
"""

import glob
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.neighbors import NearestNeighbors

from src import artefactos as art
from src.transformador import COLUMNAS_CRUDAS, CALIFICACION_MAP, TransformadorCartera, preparar_crudo


# ------------------------------------------------------------------
# Lectura y utilidades
# ------------------------------------------------------------------
def carpeta_cortes_nuevos(config):
    return os.path.join(os.path.dirname(config.data.path), "cortes_nuevos")


def leer_historico(config):
    """CSV original + todos los cortes incorporados despues."""
    partes = [pd.read_csv(config.data.path, sep=config.data.sep)]
    archivos = sorted(glob.glob(os.path.join(carpeta_cortes_nuevos(config), "*.csv")))
    partes += [pd.read_csv(a, sep=config.data.sep) for a in archivos]
    return pd.concat(partes, ignore_index=True), len(archivos)


def cargar_modelos(config):
    r = config.paths.models
    nombres = {'xgboost': 'modelo_XGBoost', 'lightgbm': 'modelo_LightGBM',
               'kmeans': 'modelo_KMeans', 'dbscan': 'modelo_DBSCAN',
               'iforest': 'modelo_IsolationForest'}
    return {k: joblib.load(os.path.join(r, f"{v}.pkl")) for k, v in nombres.items()}


def _rank01(s):
    return s.rank(pct=True, method='average')


def _shap_binario(explainer, X):
    valores, base = explainer.shap_values(X), explainer.expected_value
    if isinstance(valores, list):
        valores = valores[-1]
    if np.ndim(valores) == 3:
        valores = valores[:, :, -1]
    if np.ndim(base) > 0:
        base = np.ravel(base)[-1]
    return np.asarray(valores), float(base)


def _pendiente(g, ventana):
    g = g.sort_values('FECHA_CORTE').tail(ventana)
    if len(g) < 2:
        return np.nan
    meses = (g['FECHA_CORTE'].dt.year * 12 + g['FECHA_CORTE'].dt.month).values.astype(float)
    return float(np.polyfit(meses, g['P_principal'].values, 1)[0])


# ------------------------------------------------------------------
# Nucleo: puntuar filas crudas
# ------------------------------------------------------------------
def puntuar(df_crudo, artefactos, modelos, config):
    """
    Transforma y puntua filas crudas (31 columnas). Devuelve una fila por
    operacion y corte con P(mora) de ambos modelos y senales de los 3 detectores.
    """
    fs, fn = list(config.features_supervisado), list(config.features_nosupervisado)
    df = preparar_crudo(df_crudo)

    informe = {'filas_leidas': len(df_crudo), 'duplicadas_exactas': len(df_crudo) - len(df)}
    antes = len(df)
    df = df.drop_duplicates(['NUM_OPER', 'FECHA_CORTE'], keep='first')
    informe['clave_repetida'] = antes - len(df)
    validas = TransformadorCartera.filas_validas(df)
    informe['calificacion_fuera_norma'] = int((~validas).sum())
    df = df[validas].reset_index(drop=True)

    X_sup, X_nosup = artefactos['transformador'].transform(df)
    Xs, Xn = X_sup[fs], X_nosup[fn]

    out = df[['NUM_OPER', 'FECHA_CORTE', 'DIASMORA']].copy()
    out['TARGET'] = (out['DIASMORA'] > config.target.umbral_mora).astype(int)
    out['particion'] = out['NUM_OPER'].map(artefactos['particion']).fillna('nuevo')

    out['P_xgboost'] = modelos['xgboost'].predict_proba(Xs)[:, 1]
    out['P_lightgbm'] = modelos['lightgbm'].predict_proba(Xs)[:, 1]
    out['P_ensemble'] = (out['P_xgboost'] + out['P_lightgbm']) / 2
    out['P_principal'] = out[f"P_{config.dashboard.modelo_principal}"]

    # Isolation Forest: mas alto = mas anomalo
    out['score_if'] = -modelos['iforest'].decision_function(Xn)
    out['flag_if'] = (modelos['iforest'].predict(Xn) == -1).astype(int)
    # K-Means: pertenencia y distancia al cluster de mayor mora (fijado en train)
    km, cr = modelos['kmeans'], artefactos['cluster_riesgo_kmeans']
    out['dist_cluster_riesgo'] = np.linalg.norm(Xn.values - km.cluster_centers_[cr], axis=1)
    out['flag_kmeans'] = (km.predict(Xn) == cr).astype(int)
    # DBSCAN: fuera del radio eps de todo punto nucleo = ruido
    db = modelos['dbscan']
    if len(db.components_):
        dist = NearestNeighbors(n_neighbors=1).fit(db.components_).kneighbors(Xn.values)[0].ravel()
    else:
        dist = np.full(len(Xn), np.inf)
    out['dist_nucleo_dbscan'] = dist
    out['flag_dbscan'] = (dist > db.eps).astype(int)
    out['votos_anomalia'] = out[['flag_if', 'flag_kmeans', 'flag_dbscan']].sum(axis=1)

    out[fs] = Xs.values
    return out, informe


def _consenso(df):
    intensidad = (_rank01(df['score_if']) + _rank01(-df['dist_cluster_riesgo'])
                  + _rank01(df['dist_nucleo_dbscan'])) / 3
    return df['votos_anomalia'] + intensidad


# ------------------------------------------------------------------
# Proceso batch completo
# ------------------------------------------------------------------
def run(config, silencioso=False):
    log = (lambda *a: None) if silencioso else print
    fs = list(config.features_supervisado)
    ruta_out = os.path.join(config.paths.reports, 'scoring')
    os.makedirs(ruta_out, exist_ok=True)

    artefactos = art.cargar(config)
    modelos = cargar_modelos(config)
    crudo, n_nuevos = leer_historico(config)
    log(f"\n[Scoring] Historico: {len(crudo)} filas | cortes nuevos incorporados: {n_nuevos}")

    todo, informe = puntuar(crudo, artefactos, modelos, config)
    log(f"[Scoring] Depuracion: {informe}")

    ultimo = todo['FECHA_CORTE'].max()
    vig = todo[todo['FECHA_CORTE'] == ultimo].copy()
    ops = vig['NUM_OPER'].unique()
    historial = todo[todo['NUM_OPER'].isin(ops)].sort_values(['NUM_OPER', 'FECHA_CORTE'])

    tray = historial.groupby('NUM_OPER').apply(lambda g: pd.Series({
        'n_cortes': len(g),
        'pendiente_mensual': _pendiente(g, config.dashboard.ventana_pendiente),
    }), include_groups=False).reset_index()
    vig = vig.merge(tray, on='NUM_OPER', how='left')
    vig['en_mora'] = vig['TARGET']
    vig['score_anomalia'] = _consenso(vig)

    log("[Scoring] Calculando SHAP de la cartera vigente...")
    sx, bx = _shap_binario(shap.TreeExplainer(modelos['xgboost']), vig[fs])
    sl, bl = _shap_binario(shap.TreeExplainer(modelos['lightgbm']), vig[fs])
    for nombre, valores in [('xgboost', sx), ('lightgbm', sl)]:
        t = pd.DataFrame(valores, columns=fs)
        t.insert(0, 'NUM_OPER', vig['NUM_OPER'].values)
        t.to_csv(os.path.join(ruta_out, f'shap_{nombre}.csv'), index=False)
    pd.DataFrame({'modelo': ['xgboost', 'lightgbm'], 'base_logodds': [bx, bl]}).to_csv(
        os.path.join(ruta_out, 'shap_base.csv'), index=False)

    historial[['NUM_OPER', 'FECHA_CORTE', 'particion', 'DIASMORA', 'TARGET',
               'P_xgboost', 'P_lightgbm', 'P_ensemble']].to_csv(
        os.path.join(ruta_out, 'historial_puntuado.csv'), index=False)
    vig.to_csv(os.path.join(ruta_out, 'cartera_vigente.csv'), index=False)
    pd.DataFrame([{
        'ultimo_corte': ultimo.strftime('%Y-%m-%d'),
        'corte_entrenamiento': artefactos['corte_entrenamiento'].strftime('%Y-%m-%d'),
        'operaciones_vigentes': len(vig),
        'operaciones_nuevas': int((vig['particion'] == 'nuevo').sum()),
        'en_mora': int(vig['en_mora'].sum()),
        'cortes_nuevos_incorporados': n_nuevos,
        'modelo_principal': config.dashboard.modelo_principal,
        'generado': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }]).to_csv(os.path.join(ruta_out, 'metadata.csv'), index=False)

    log(f"[Scoring] Ultimo corte: {ultimo:%d/%m/%Y} | Vigentes: {len(vig)} | "
        f"Nuevas: {int((vig['particion'] == 'nuevo').sum())} | En mora: {int(vig['en_mora'].sum())} | "
        f"Con trayectoria: {int((vig['n_cortes'] >= 2).sum())}")
    log(f"[Scoring] Anomalias por votos: {vig['votos_anomalia'].value_counts().sort_index().to_dict()}")
    log(f"[Scoring] Archivos listos en: {ruta_out}")
    return vig


# ------------------------------------------------------------------
# Funciones para cortes nuevos (dashboard y consola)
# ------------------------------------------------------------------
def validar_corte(df_nuevo, config):
    """Devuelve (errores, avisos). Con errores no se puede incorporar."""
    errores, avisos = [], []
    faltan = [c for c in COLUMNAS_CRUDAS if c not in df_nuevo.columns]
    sobran = [c for c in df_nuevo.columns if c not in COLUMNAS_CRUDAS]
    if faltan:
        errores.append(f"Faltan columnas: {faltan}")
    if sobran:
        avisos.append(f"Columnas extra que se ignoraran: {sobran}")
    if faltan:
        return errores, avisos

    try:
        fechas = pd.to_datetime(df_nuevo['FECHA_CORTE'], format='%d/%m/%Y')
    except (ValueError, TypeError):
        errores.append("FECHA_CORTE no tiene el formato dia/mes/anio (ej. 31/1/2020).")
        return errores, avisos
    if fechas.nunique() != 1:
        errores.append(f"El archivo debe tener un solo corte; tiene {fechas.nunique()}.")
        return errores, avisos

    crudo, _ = leer_historico(config)
    ultimo_hist = pd.to_datetime(crudo['FECHA_CORTE'], format='%d/%m/%Y').max()
    corte = fechas.iloc[0]
    if corte <= ultimo_hist:
        errores.append(f"El corte {corte:%d/%m/%Y} no es posterior al ultimo del historico "
                       f"({ultimo_hist:%d/%m/%Y}). Posible archivo repetido.")
    fuera = ~df_nuevo['CALIFICACIONCOD'].isin(list(CALIFICACION_MAP))
    if fuera.any():
        avisos.append(f"{int(fuera.sum())} filas con calificacion fuera de norma se excluiran.")
    return errores, avisos


def puntuar_corte_nuevo(df_nuevo, config):
    """Vista previa: puntua solo el corte nuevo, sin guardar nada."""
    artefactos = art.cargar(config)
    modelos = cargar_modelos(config)
    res, informe = puntuar(df_nuevo[COLUMNAS_CRUDAS], artefactos, modelos, config)
    res['score_anomalia'] = _consenso(res)
    return res, informe


def incorporar_corte(df_nuevo, config):
    """Guarda el corte en data/raw/cortes_nuevos/ y regenera todo el scoring."""
    fecha = pd.to_datetime(df_nuevo['FECHA_CORTE'].iloc[0], format='%d/%m/%Y')
    carpeta = carpeta_cortes_nuevos(config)
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"corte_{fecha:%Y_%m}.csv")
    df_nuevo[COLUMNAS_CRUDAS].to_csv(ruta, sep=config.data.sep, index=False)
    run(config, silencioso=True)
    return ruta
