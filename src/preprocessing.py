"""
preprocessing.py
================
Pipeline completo de preprocesamiento para behavioral scoring.
Fiel al QMD original: misma logica, mismos parametros, sin hardcodeo.

Flujo:
  CSV crudo
    -> limpieza basica (duplicados, fechas, derivadas)
    -> construccion del TARGET (DIASMORA > umbral)
    -> eliminacion de multicolinealidad perfecta
    -> particion por operacion (StratifiedGroupKFold en NUM_OPER)
    -> transformaciones supervisadas  (log1p + StandardScaler / StandardScaler / log1p)
    -> transformaciones no supervisadas (log1p + MinMaxScaler  / MinMaxScaler  / log1p)
    -> eliminacion de constantes
    -> depuracion de calificaciones fuera de norma (A, B genericos)
    -> codificacion categorica
    -> SMOTE (solo en entrenamiento supervisado)
    -> devuelve diccionario con todos los datasets + scalers
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE


# ============================================================
# MAPA ORDINAL DE CALIFICACION SEPS
# ============================================================
CALIFICACION_MAP = {
    'A1': 0, 'A2': 1, 'A3': 2,
    'B1': 3, 'B2': 4,
    'C1': 5, 'C2': 6,
    'D': 7, 'E': 8
}


def run_preprocessing(config):
    """
    Ejecuta el pipeline completo de preprocesamiento.

    Parametros:
        config : DictConfig de Hydra con toda la configuracion.

    Retorna:
        dict con claves:
            X_train_sup, X_val_sup, X_test_sup
            X_train_sup_bal, y_train_bal
            y_train, y_val, y_test
            X_train_nosup, X_val_nosup, X_test_nosup
            scalers_sup  : dict {nombre_var: scaler} para inferencia
            scalers_nosup: dict {nombre_var: scaler} para inferencia
            ohe_sup      : OneHotEncoder ajustado en train sup
            ohe_nosup    : OneHotEncoder ajustado en train nosup
    """

    ruta_csv = config.data.path
    sep      = config.data.sep
    umbral   = config.target.umbral_mora
    rs       = config.split.random_state

    # ----------------------------------------------------------------
    # 1. CARGA
    # ----------------------------------------------------------------
    print(f"\n[Preprocessing] Leyendo CSV: {ruta_csv}")
    base = pd.read_csv(ruta_csv, sep=sep)
    print(f"   Shape original: {base.shape}")

    # ----------------------------------------------------------------
    # 2. CONVERSION DE FECHAS Y DERIVADAS TEMPORALES
    # ----------------------------------------------------------------
    cols_fecha = ['FECHA_CORTE', 'FECHACONCESION', 'FECHAVENCIMIENTO']
    for col in cols_fecha:
        base[col] = pd.to_datetime(base[col], format='%d/%m/%Y')

    base['DIAS_MADURACION'] = (
        base['FECHA_CORTE'] - base['FECHACONCESION']
    ).dt.days.astype('int64')

    base['DIAS_RESTANTES'] = (
        base['FECHAVENCIMIENTO'] - base['FECHA_CORTE']
    ).dt.days.astype('int64')

    # ----------------------------------------------------------------
    # 3. ELIMINACION DE DUPLICADOS
    # ----------------------------------------------------------------
    antes = len(base)
    base = base.drop_duplicates()
    despues = len(base)
    print(f"   Duplicados eliminados: {antes - despues} | Filas validas: {despues}")

    # ----------------------------------------------------------------
    # 4. TARGET: DIASMORA > umbral -> 1
    # ----------------------------------------------------------------
    base['TARGET'] = (base['DIASMORA'] > umbral).astype(int)
    print(f"   Tasa de mora (TARGET=1): {base['TARGET'].mean():.4f}")

    # ----------------------------------------------------------------
    # 5. ELIMINAR MULTICOLINEALIDAD PERFECTA
    # ----------------------------------------------------------------
    base = base.drop(columns=['SALDOTOTAL', 'PLAZO_ORIGINAL_MESES'])

    # ----------------------------------------------------------------
    # 6. SEPARAR X e y — NUM_OPER como indice de grupo
    # ----------------------------------------------------------------
    base = base.set_index('NUM_OPER')
    fechas_corte = base['FECHA_CORTE']   # se conserva aparte para trayectorias
    X = base.drop(columns=['TARGET', 'FECHA_CORTE', 'FECHACONCESION', 'FECHAVENCIMIENTO'])
    y = base['TARGET']

    # ----------------------------------------------------------------
    # 7. PARTICION POR OPERACION (anti-fuga de datos)
    #    StratifiedGroupKFold: cada operacion va completa a un solo split
    # ----------------------------------------------------------------
    groups = X.index  # NUM_OPER como grupo

    # Paso 1: 70% train vs 30% temp
    sgkf1 = StratifiedGroupKFold(
        n_splits=config.split.n_splits_train,
        shuffle=True,
        random_state=rs
    )
    train_idx, temp_idx = next(sgkf1.split(X, y, groups=groups))

    X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
    y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]
    groups_temp     = pd.Series(groups).iloc[temp_idx]

    # Paso 2: 50% val vs 50% test del 30% restante
    sgkf2 = StratifiedGroupKFold(
        n_splits=config.split.n_splits_val_test,
        shuffle=True,
        random_state=rs
    )
    val_idx_rel, test_idx_rel = next(
        sgkf2.split(X_temp, y_temp, groups=groups_temp)
    )

    X_val,  X_test  = X_temp.iloc[val_idx_rel],  X_temp.iloc[test_idx_rel]

    # Fechas de corte alineadas por posicion con cada particion
    f_train = fechas_corte.iloc[train_idx]
    f_temp  = fechas_corte.iloc[temp_idx]
    f_val, f_test = f_temp.iloc[val_idx_rel], f_temp.iloc[test_idx_rel]
    y_val,  y_test  = y_temp.iloc[val_idx_rel],   y_temp.iloc[test_idx_rel]

    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"   Interseccion train-val:  {len(set(X_train.index) & set(X_val.index))}")
    print(f"   Interseccion train-test: {len(set(X_train.index) & set(X_test.index))}")

    # ----------------------------------------------------------------
    # 8. COPIAS PARA RUTAS SUPERVISADA Y NO SUPERVISADA
    # ----------------------------------------------------------------
    X_train_sup   = X_train.copy()
    X_val_sup     = X_val.copy()
    X_test_sup    = X_test.copy()

    X_train_nosup = X_train.copy()
    X_val_nosup   = X_val.copy()
    X_test_nosup  = X_test.copy()

    # ----------------------------------------------------------------
    # 9. TRANSFORMACIONES SUPERVISADAS
    #    Cada variable tiene su propio scaler ajustado en train
    #    (fiel al QMD que reutiliza el mismo objeto con fit_transform)
    # ----------------------------------------------------------------
    scalers_sup = {}

    # Grupo A: log(1+x) + StandardScaler
    vars_log_est = [
        'MONTOCREDITO', 'MONTOCUOTA', 'SALDOCAPITAL_POR_VENCER',
        'SALDOCAPITAL_ND', 'SALDOCAPITAL_JUDICIAL',
        'NUMEROVECESMORA_60', 'NUMEROVECESMORA_90', 'NUMEROVECESMORA_180',
        'NUMEROVECESMORA_360', 'NUMERORENEGOCIACION'
    ]
    for var in vars_log_est:
        s = StandardScaler()
        suffix = f'{var}_log_est'
        X_train_sup[suffix] = s.fit_transform(np.log1p(X_train[[var]]))
        X_val_sup[suffix]   = s.transform(np.log1p(X_val[[var]]))
        X_test_sup[suffix]  = s.transform(np.log1p(X_test[[var]]))
        scalers_sup[suffix] = s

    # Grupo B: solo StandardScaler
    vars_est = ['PLAZO', 'TASAINTERES', 'DIAS_MADURACION', 'DIAS_RESTANTES']
    for var in vars_est:
        s = StandardScaler()
        suffix = f'{var}_est'
        X_train_sup[suffix] = s.fit_transform(X_train[[var]])
        X_val_sup[suffix]   = s.transform(X_val[[var]])
        X_test_sup[suffix]  = s.transform(X_test[[var]])
        scalers_sup[suffix] = s

    # Grupo C: solo log(1+x) — variables clave de mora (sin escalar para preservar señal)
    vars_log = [
        'SALDOCAPITAL_VENCIDO', 'DIASMORA', 'MORAMAXIMA', 'MORAPROMEDIO',
        'NUMEROVECESMORA_30', 'NUMEROVECESMORA_MAS360', 'SALDOCASTIGADO'
    ]
    for var in vars_log:
        suffix = f'{var}_log'
        X_train_sup[suffix] = np.log1p(X_train[[var]]).values
        X_val_sup[suffix]   = np.log1p(X_val[[var]]).values
        X_test_sup[suffix]  = np.log1p(X_test[[var]]).values

    # ----------------------------------------------------------------
    # 10. TRANSFORMACIONES NO SUPERVISADAS
    # ----------------------------------------------------------------
    scalers_nosup = {}

    # Grupo A: log(1+x) + MinMaxScaler
    vars_log_norm = [
        'MONTOCREDITO', 'MONTOCUOTA', 'SALDOCAPITAL_POR_VENCER',
        'SALDOCAPITAL_ND', 'SALDOCAPITAL_JUDICIAL',
        'NUMEROVECESMORA_60', 'NUMEROVECESMORA_90', 'NUMEROVECESMORA_180',
        'NUMEROVECESMORA_360', 'NUMERORENEGOCIACION'
    ]
    for var in vars_log_norm:
        s = MinMaxScaler()
        suffix = f'{var}_log_norm'
        X_train_nosup[suffix] = s.fit_transform(np.log1p(X_train[[var]]))
        X_val_nosup[suffix]   = s.transform(np.log1p(X_val[[var]]))
        X_test_nosup[suffix]  = s.transform(np.log1p(X_test[[var]]))
        scalers_nosup[suffix] = s

    # Grupo B: solo MinMaxScaler
    vars_norm = ['PLAZO', 'TASAINTERES', 'DIAS_MADURACION', 'DIAS_RESTANTES']
    for var in vars_norm:
        s = MinMaxScaler()
        suffix = f'{var}_norm'
        X_train_nosup[suffix] = s.fit_transform(X_train[[var]])
        X_val_nosup[suffix]   = s.transform(X_val[[var]])
        X_test_nosup[suffix]  = s.transform(X_test[[var]])
        scalers_nosup[suffix] = s

    # Grupo C: solo log(1+x)
    vars_log_nosup = [
        'SALDOCAPITAL_VENCIDO', 'DIASMORA', 'MORAMAXIMA', 'MORAPROMEDIO',
        'NUMEROVECESMORA_30', 'NUMEROVECESMORA_MAS360', 'SALDOCASTIGADO'
    ]
    for var in vars_log_nosup:
        suffix = f'{var}_log'
        X_train_nosup[suffix] = np.log1p(X_train[[var]]).values
        X_val_nosup[suffix]   = np.log1p(X_val[[var]]).values
        X_test_nosup[suffix]  = np.log1p(X_test[[var]]).values

    # ----------------------------------------------------------------
    # 11. ELIMINAR CONSTANTES
    # ----------------------------------------------------------------
    constantes = ['TIPOPERSONACOD', 'TIPOCREDITOCOD', 'SALDOCASTIGADO', 'SALDOCASTIGADO_log']
    for dfs in [
        [X_train_sup, X_val_sup, X_test_sup],
        [X_train_nosup, X_val_nosup, X_test_nosup]
    ]:
        for df in dfs:
            cols_a_drop = [c for c in constantes if c in df.columns]
            df.drop(columns=cols_a_drop, inplace=True)

    # ----------------------------------------------------------------
    # 12. DEPURACION DE CALIFICACIONES FUERA DE NORMA (A, B genericos)
    # ----------------------------------------------------------------
    mask_train = ~X_train_sup['CALIFICACIONCOD'].isin(['A', 'B'])
    X_train_sup   = X_train_sup[mask_train].copy()
    X_train_nosup = X_train_nosup[mask_train].copy()
    y_train       = y_train[mask_train].copy()
    f_train       = f_train[mask_train.values].copy()

    mask_val = ~X_val_sup['CALIFICACIONCOD'].isin(['A', 'B'])
    X_val_sup   = X_val_sup[mask_val].copy()
    X_val_nosup = X_val_nosup[mask_val].copy()
    y_val       = y_val[mask_val].copy()
    f_val       = f_val[mask_val.values].copy()

    mask_test = ~X_test_sup['CALIFICACIONCOD'].isin(['A', 'B'])
    X_test_sup   = X_test_sup[mask_test].copy()
    X_test_nosup = X_test_nosup[mask_test].copy()
    y_test       = y_test[mask_test].copy()
    f_test       = f_test[mask_test.values].copy()

    print(f"   Tras depuracion A/B — Train: {len(X_train_sup)} | Val: {len(X_val_sup)} | Test: {len(X_test_sup)}")

    # ----------------------------------------------------------------
    # 13. CODIFICACION CATEGORICA
    # ----------------------------------------------------------------

    # 13a. PERIOD_PAGO: agrupar no-MENS como OTROS, luego binario
    for df in [X_train_sup, X_val_sup, X_test_sup,
               X_train_nosup, X_val_nosup, X_test_nosup]:
        df['PERIOD_PAGO'] = df['PERIOD_PAGO'].apply(
            lambda x: x if x == 'MENS' else 'OTROS'
        )
        df['PERIOD_PAGO_OTROS_e'] = (df['PERIOD_PAGO'] == 'OTROS').astype(float)
        df.drop(columns=['PERIOD_PAGO'], inplace=True)

    # 13b. CALIFICACIONCOD: ordinal segun norma SEPS
    for df in [X_train_sup, X_val_sup, X_test_sup,
               X_train_nosup, X_val_nosup, X_test_nosup]:
        df['CALIFICACIONCOD'] = df['CALIFICACIONCOD'].map(CALIFICACION_MAP)
        df.rename(columns={'CALIFICACIONCOD': 'CALIFICACIONCOD_e'}, inplace=True)

    # 13c. PRODUCTO_COD + TIPOTASACOD: OHE con drop='if_binary'
    #      Encoder ajustado SOLO en X_train para evitar fuga
    ohe_sup = OneHotEncoder(drop='if_binary', sparse_output=False)
    ohe_sup.fit(X_train_sup[['PRODUCTO_COD', 'TIPOTASACOD']])

    for df in [X_train_sup, X_val_sup, X_test_sup]:
        enc = ohe_sup.transform(df[['PRODUCTO_COD', 'TIPOTASACOD']])
        orig_names = ohe_sup.get_feature_names_out(['PRODUCTO_COD', 'TIPOTASACOD'])
        new_names  = [c + '_e' for c in orig_names]
        df[new_names] = pd.DataFrame(enc, columns=new_names, index=df.index)
        df.drop(columns=['PRODUCTO_COD', 'TIPOTASACOD'], inplace=True)

    ohe_nosup = OneHotEncoder(drop='if_binary', sparse_output=False)
    ohe_nosup.fit(X_train_nosup[['PRODUCTO_COD', 'TIPOTASACOD']])

    for df in [X_train_nosup, X_val_nosup, X_test_nosup]:
        enc = ohe_nosup.transform(df[['PRODUCTO_COD', 'TIPOTASACOD']])
        orig_names = ohe_nosup.get_feature_names_out(['PRODUCTO_COD', 'TIPOTASACOD'])
        new_names  = [c + '_e' for c in orig_names]
        df[new_names] = pd.DataFrame(enc, columns=new_names, index=df.index)
        df.drop(columns=['PRODUCTO_COD', 'TIPOTASACOD'], inplace=True)

    print(f"   Columnas codificadas (_e): {[c for c in X_train_sup.columns if c.endswith('_e')]}")
    print(f"   Shape final supervisado:     {X_train_sup.shape}")
    print(f"   Shape final no supervisado:  {X_train_nosup.shape}")

    # ----------------------------------------------------------------
    # 14. SMOTE — solo en entrenamiento supervisado
    #     Val y Test quedan intactos (proporciones reales de la cartera)
    #     No supervisados no usan SMOTE (distorsionaria la deteccion)
    # ----------------------------------------------------------------
    smote = SMOTE(random_state=config.smote.random_state)
    X_train_sup_bal, y_train_bal = smote.fit_resample(X_train_sup, y_train)

    print(f"   SMOTE — Train original: {X_train_sup.shape[0]} | Balanceado: {X_train_sup_bal.shape[0]}")
    print(f"   Distribucion balanceada: {dict(pd.Series(y_train_bal).value_counts())}")

    # ----------------------------------------------------------------
    # 15. RETORNO
    # ----------------------------------------------------------------
    return {
        # Supervisado
        'X_train_sup':     X_train_sup,
        'X_val_sup':       X_val_sup,
        'X_test_sup':      X_test_sup,
        'X_train_sup_bal': X_train_sup_bal,
        'y_train':         y_train,
        'y_val':           y_val,
        'y_test':          y_test,
        'y_train_bal':     y_train_bal,
        # No supervisado
        'X_train_nosup':   X_train_nosup,
        'X_val_nosup':     X_val_nosup,
        'X_test_nosup':    X_test_nosup,
        # Scalers y encoders para inferencia
        'scalers_sup':     scalers_sup,
        'scalers_nosup':   scalers_nosup,
        'ohe_sup':         ohe_sup,
        'ohe_nosup':       ohe_nosup,
        # Fechas de corte por fila (para trayectorias en scoring de cartera)
        'fecha_train':     f_train,
        'fecha_val':       f_val,
        'fecha_test':      f_test,
    }
