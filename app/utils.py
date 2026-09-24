"""
utils.py — funciones compartidas por todas las paginas del dashboard.
El dashboard solo LEE los CSV de reports/scoring/ (generados por score_cartera.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
RUTA_SCORING = ROOT / "reports" / "scoring"
if str(ROOT) not in sys.path:          # permite importar src/ desde el tablero
    sys.path.insert(0, str(ROOT))

# Paleta profesional
AZUL = "#0B5C8C"
AZUL_OSCURO = "#0B2545"
VERDE = "#2E9E6B"
AMBAR = "#E0A526"
NARANJA = "#E07B26"
ROJO = "#C8413A"
GRIS = "#6B7A8C"

NIVELES = ["Sin alerta", "Riesgo bajo", "Riesgo moderado",
           "Riesgo alto", "Riesgo muy alto", "Riesgo critico"]
COLORES_NIVEL = [VERDE, "#9BBF3B", AMBAR, NARANJA, ROJO, "#8E1F1A"]

NOMBRES = {
    "MONTOCREDITO_log_est": "Monto del credito",
    "MONTOCUOTA_log_est": "Monto de la cuota",
    "SALDOCAPITAL_POR_VENCER_log_est": "Saldo por vencer",
    "NUMERORENEGOCIACION_log_est": "Numero de renegociaciones",
    "PLAZO_est": "Plazo",
    "TASAINTERES_est": "Tasa de interes",
    "DIAS_MADURACION_est": "Dias de maduracion",
    "DIAS_RESTANTES_est": "Dias restantes",
    "PRODUCTO_COD_PROD_2_e": "Producto 2",
    "TIPOTASACOD_V_e": "Tasa variable",
    "PERIOD_PAGO_OTROS_e": "Pago no mensual",
}
ETIQUETA_MODELO = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "ensemble": "Ensemble"}


# ------------------------------------------------------------------
# Carga
# ------------------------------------------------------------------
@st.cache_data
def cargar_config():
    return OmegaConf.to_container(OmegaConf.load(ROOT / "config" / "main.yaml"))


@st.cache_data
def cargar_datos():
    archivos = ["cartera_vigente", "historial_puntuado", "shap_xgboost",
                "shap_lightgbm", "shap_base", "metadata"]
    faltan = [a for a in archivos if not (RUTA_SCORING / f"{a}.csv").exists()]
    if faltan:
        return None
    d = {a: pd.read_csv(RUTA_SCORING / f"{a}.csv") for a in archivos}
    d["cartera_vigente"]["FECHA_CORTE"] = pd.to_datetime(d["cartera_vigente"]["FECHA_CORTE"])
    d["historial_puntuado"]["FECHA_CORTE"] = pd.to_datetime(d["historial_puntuado"]["FECHA_CORTE"])
    return d


def datos_o_detener():
    d = cargar_datos()
    if d is None:
        st.error("No se encontraron los archivos de scoring en `reports/scoring/`.")
        st.info("Ejecuta en consola, desde la raiz del proyecto:\n\n"
                "`python score_cartera.py`\n\ny luego recarga esta pagina.")
        st.stop()
    return d


# ------------------------------------------------------------------
# Estilo y componentes
# ------------------------------------------------------------------
def aplicar_estilo(titulo, subtitulo):
    st.markdown(f"""
    <style>
      .block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
      .encabezado {{background: linear-gradient(90deg, {AZUL_OSCURO} 0%, {AZUL} 100%);
                   color: white; padding: 18px 24px; border-radius: 10px; margin-bottom: 18px;}}
      .encabezado h1 {{color: white; font-size: 1.55rem; margin: 0;}}
      .encabezado p {{color: #D6E4F0; margin: 4px 0 0 0; font-size: 0.95rem;}}
      .kpi {{background: white; border-radius: 10px; padding: 14px 16px;
             border-left: 5px solid {AZUL}; box-shadow: 0 1px 3px rgba(0,0,0,0.08);}}
      .kpi .lbl {{color: {GRIS}; font-size: 0.80rem; text-transform: uppercase; letter-spacing: .04em;}}
      .kpi .val {{color: {AZUL_OSCURO}; font-size: 1.65rem; font-weight: 700; line-height: 1.2;}}
      .kpi .sub {{color: {GRIS}; font-size: 0.80rem;}}
      .semaforo {{border-radius: 10px; padding: 16px; color: white; text-align: center;}}
      .semaforo .n {{font-size: 1.35rem; font-weight: 700;}}
      div[data-testid="stSidebar"] {{background-color: {AZUL_OSCURO};}}
      div[data-testid="stSidebar"] * {{color: #E8EEF4;}}
    </style>
    <div class="encabezado"><h1>{titulo}</h1><p>{subtitulo}</p></div>
    """, unsafe_allow_html=True)


def kpi(col, etiqueta, valor, sub="", color=AZUL):
    col.markdown(f"""<div class="kpi" style="border-left-color:{color}">
        <div class="lbl">{etiqueta}</div><div class="val">{valor}</div>
        <div class="sub">{sub}</div></div>""", unsafe_allow_html=True)


def barra_lateral(datos):
    """Controles comunes: modelo y umbral. Se guardan en session_state entre paginas."""
    cfg = cargar_config()["dashboard"]
    meta = datos["metadata"].iloc[0]
    if "modelo" not in st.session_state:
        st.session_state.modelo = cfg["modelo_principal"]
    if "umbral" not in st.session_state:
        st.session_state.umbral = cfg["umbral_defecto"]

    with st.sidebar:
        st.markdown("### Behavioral Scoring")
        st.caption(f"Corte analizado: **{pd.to_datetime(meta['ultimo_corte']):%d/%m/%Y}**")
        st.caption(f"Scoring generado: {meta['generado']}")
        st.divider()
        st.session_state.modelo = st.selectbox(
            "Modelo de probabilidad", list(ETIQUETA_MODELO),
            index=list(ETIQUETA_MODELO).index(st.session_state.modelo),
            format_func=ETIQUETA_MODELO.get)
        st.session_state.umbral = st.select_slider(
            "Umbral de incumplimiento", options=cfg["umbrales"],
            value=st.session_state.umbral, format_func=lambda x: f"{x:.2f}")
        st.divider()
        st.caption("XGBoost: mejor F1 (equilibrio precision/recall). "
                   "LightGBM: mejor calibracion (Brier).")
    return st.session_state.modelo, st.session_state.umbral


def config_absoluta():
    """Config con rutas absolutas para ejecutar el motor de scoring desde el tablero."""
    from src.config_local import cargar_config as _c
    return _c()


def refrescar_datos():
    """Tras incorporar un corte: vaciar cache para que todas las paginas relean los CSV."""
    st.cache_data.clear()
    st.cache_resource.clear()


def marca(v):
    return "✔" if int(v) == 1 else "—"


def col_p(modelo):
    return f"P_{modelo}"


def nivel_riesgo(p, umbrales):
    """Cuantos umbrales supera la probabilidad (0 = sin alerta)."""
    return int(sum(p >= u for u in umbrales))


def tabla_descarga(df, nombre, etiqueta="Descargar tabla (CSV)"):
    st.download_button(etiqueta, df.to_csv(index=False).encode("utf-8"),
                       file_name=nombre, mime="text/csv", width="content")


def config_columnas_p():
    return {
        "P_xgboost": st.column_config.ProgressColumn("P XGBoost", format="%.3f", min_value=0, max_value=1),
        "P_lightgbm": st.column_config.ProgressColumn("P LightGBM", format="%.3f", min_value=0, max_value=1),
        "P_ensemble": st.column_config.ProgressColumn("P Ensemble", format="%.3f", min_value=0, max_value=1),
        "NUM_OPER": st.column_config.NumberColumn("Operacion", format="%d"),
        "DIASMORA": st.column_config.NumberColumn("Dias mora", format="%d"),
        "particion": st.column_config.TextColumn("Particion",
                                                 help="train/val/test del entrenamiento · nuevo = operacion posterior"),
    }
