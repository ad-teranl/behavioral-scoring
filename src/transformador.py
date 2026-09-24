"""
transformador.py
================
TransformadorCartera: aplica a CUALQUIER corte (historico o nuevo) exactamente
las mismas transformaciones del entrenamiento 2017-2019, sin volver a aprender.

  - Escaladores (StandardScaler / MinMaxScaler) y One-Hot: se reciben YA AJUSTADOS
    desde preprocessing.run_preprocessing() -> nunca se hace fit() con datos nuevos.
  - log1p, mapa ordinal SEPS y regla PERIOD_PAGO: son reglas fijas.

Se guarda una sola vez en models/artefactos_scoring.pkl y luego solo se aplica.
"""

import numpy as np
import pandas as pd

COLUMNAS_CRUDAS = [
    'NUM_OPER', 'FECHA_CORTE', 'PRODUCTO_COD', 'TIPOPERSONACOD', 'MONTOCREDITO',
    'FECHACONCESION', 'FECHAVENCIMIENTO', 'PLAZO_ORIGINAL_MESES', 'PLAZO', 'MONTOCUOTA',
    'PERIOD_PAGO', 'TIPOTASACOD', 'TASAINTERES', 'TIPOCREDITOCOD', 'CALIFICACIONCOD',
    'SALDOCAPITAL_POR_VENCER', 'SALDOCAPITAL_VENCIDO', 'SALDOCAPITAL_ND',
    'SALDOCAPITAL_JUDICIAL', 'SALDOCASTIGADO', 'SALDOTOTAL', 'DIASMORA', 'MORAMAXIMA',
    'MORAPROMEDIO', 'NUMERORENEGOCIACION', 'NUMEROVECESMORA_30', 'NUMEROVECESMORA_60',
    'NUMEROVECESMORA_90', 'NUMEROVECESMORA_180', 'NUMEROVECESMORA_360',
    'NUMEROVECESMORA_MAS360',
]

VARS_LOG_ESCALA = [  # log1p + escalador (Standard en sup, MinMax en nosup)
    'MONTOCREDITO', 'MONTOCUOTA', 'SALDOCAPITAL_POR_VENCER', 'SALDOCAPITAL_ND',
    'SALDOCAPITAL_JUDICIAL', 'NUMEROVECESMORA_60', 'NUMEROVECESMORA_90',
    'NUMEROVECESMORA_180', 'NUMEROVECESMORA_360', 'NUMERORENEGOCIACION',
]
VARS_ESCALA = ['PLAZO', 'TASAINTERES', 'DIAS_MADURACION', 'DIAS_RESTANTES']
VARS_LOG = ['SALDOCAPITAL_VENCIDO', 'DIASMORA', 'MORAMAXIMA', 'MORAPROMEDIO',
            'NUMEROVECESMORA_30', 'NUMEROVECESMORA_MAS360']

CALIFICACION_MAP = {'A1': 0, 'A2': 1, 'A3': 2, 'B1': 3, 'B2': 4,
                    'C1': 5, 'C2': 6, 'D': 7, 'E': 8}


def preparar_crudo(df):
    """Limpieza comun a todo corte: fechas, variables derivadas y duplicados."""
    df = df.copy()
    for col in ['FECHA_CORTE', 'FECHACONCESION', 'FECHAVENCIMIENTO']:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y')
    df['DIAS_MADURACION'] = (df['FECHA_CORTE'] - df['FECHACONCESION']).dt.days.astype('int64')
    df['DIAS_RESTANTES'] = (df['FECHAVENCIMIENTO'] - df['FECHA_CORTE']).dt.days.astype('int64')
    return df.drop_duplicates()


class TransformadorCartera:
    """Contenedor de escaladores/codificadores ajustados en train + reglas fijas."""

    def __init__(self, scalers_sup, scalers_nosup, ohe_sup, ohe_nosup):
        self.scalers_sup = scalers_sup
        self.scalers_nosup = scalers_nosup
        self.ohe_sup = ohe_sup
        self.ohe_nosup = ohe_nosup

    # ---------------------------------------------------------------
    def _categoricas(self, df, X, ohe):
        X['PERIOD_PAGO_OTROS_e'] = (df['PERIOD_PAGO'] != 'MENS').astype(float).values
        X['CALIFICACIONCOD_e'] = df['CALIFICACIONCOD'].map(CALIFICACION_MAP).values
        cats = df[['PRODUCTO_COD', 'TIPOTASACOD']]
        enc = ohe.transform(cats)
        nombres = [c + '_e' for c in ohe.get_feature_names_out(['PRODUCTO_COD', 'TIPOTASACOD'])]
        for i, n in enumerate(nombres):
            X[n] = enc[:, i]
        return X

    def transform(self, df):
        """
        Recibe un DataFrame ya pasado por preparar_crudo().
        Devuelve (X_sup, X_nosup) con TODAS las variables transformadas,
        indexadas igual que df. Solo aplica transform(), nunca fit().
        """
        X_sup = pd.DataFrame(index=df.index)
        X_nosup = pd.DataFrame(index=df.index)

        for var in VARS_LOG_ESCALA:
            logv = np.log1p(df[[var]])
            X_sup[f'{var}_log_est'] = self.scalers_sup[f'{var}_log_est'].transform(logv).ravel()
            X_nosup[f'{var}_log_norm'] = self.scalers_nosup[f'{var}_log_norm'].transform(logv).ravel()
        for var in VARS_ESCALA:
            X_sup[f'{var}_est'] = self.scalers_sup[f'{var}_est'].transform(df[[var]]).ravel()
            X_nosup[f'{var}_norm'] = self.scalers_nosup[f'{var}_norm'].transform(df[[var]]).ravel()
        for var in VARS_LOG:
            X_sup[f'{var}_log'] = np.log1p(df[var]).values
            X_nosup[f'{var}_log'] = np.log1p(df[var]).values

        X_sup = self._categoricas(df, X_sup, self.ohe_sup)
        X_nosup = self._categoricas(df, X_nosup, self.ohe_nosup)
        return X_sup, X_nosup

    @staticmethod
    def filas_validas(df):
        """Calificaciones fuera de norma SEPS (A, B genericos) no se pueden codificar."""
        return df['CALIFICACIONCOD'].isin(CALIFICACION_MAP.keys())
