"""
Cargar nuevo corte — sube el CSV mensual de la cooperativa (31 columnas originales):
  1. Valida formato y que sea posterior al historico
  2. Vista previa: P(mora) y senales de cada detector, SIN guardar nada
  3. Boton para incorporarlo al historico y regenerar todo el tablero
Los escaladores y modelos NO se reajustan: se usan los congelados del entrenamiento.
"""
import glob
import io
import os

import pandas as pd
import streamlit as st

from app import utils as u

st.set_page_config(page_title="Behavioral Scoring | Cargar corte", page_icon="📥", layout="wide")
datos = u.datos_o_detener()
modelo, umbral = u.barra_lateral(datos)
vig_actual = datos["cartera_vigente"]
P = u.col_p(modelo)

u.aplicar_estilo("Cargar nuevo corte mensual",
                 "Sube el archivo de la cooperativa, revisa los resultados y decide si incorporarlo")

from src import scoring_cartera as sc  # noqa: E402  (requiere ROOT en sys.path, lo hace utils)

config = u.config_absoluta()


@st.cache_data(show_spinner="Validando y puntuando el corte...")
def previsualizar(contenido: bytes):
    df = pd.read_csv(io.BytesIO(contenido), sep=";", encoding="utf-8-sig")
    if df.shape[1] == 1:                       # por si viene separado por comas
        df = pd.read_csv(io.BytesIO(contenido), sep=",", encoding="utf-8-sig")
    errores, avisos = sc.validar_corte(df, config)
    if errores:
        return df, errores, avisos, None, None
    res, informe = sc.puntuar_corte_nuevo(df, config)
    return df, errores, avisos, res, informe


archivo = st.file_uploader("Archivo CSV del corte (separado por punto y coma, mismas 31 columnas)",
                           type=["csv"])

if archivo is None:
    st.info("Aun no hay archivo. Para probar, usa `data/ejemplos/corte_2020_01_SIMULADO.csv` "
            "(589 operaciones de diciembre 2019 actualizadas + 5 nuevas).")
else:
    df, errores, avisos, res, informe = previsualizar(archivo.getvalue())
    for a in avisos:
        st.warning(a)
    if errores:
        for e in errores:
            st.error(e)
        st.stop()

    corte = pd.to_datetime(df["FECHA_CORTE"].iloc[0], format="%d/%m/%Y")
    previas = set(vig_actual.NUM_OPER)
    actuales = set(res.NUM_OPER)
    nuevas = res[res.particion == "nuevo"]

    c1, c2, c3, c4, c5 = st.columns(5)
    u.kpi(c1, "Corte del archivo", f"{corte:%d/%m/%Y}", f"{len(df):,} filas leidas")
    u.kpi(c2, "Continuan", f"{len(actuales & previas):,}", "ya estaban vigentes", u.AZUL)
    u.kpi(c3, "Operaciones nuevas", f"{len(nuevas):,}", "nunca vistas por los modelos", u.VERDE)
    u.kpi(c4, "Salieron", f"{len(previas - actuales):,}", "canceladas o castigadas", u.GRIS)
    u.kpi(c5, "En mora en el corte", f"{int(res.TARGET.sum()):,}",
          f"{res.TARGET.mean()*100:.1f}% del corte", u.ROJO)
    if sum(informe[k] for k in ["duplicadas_exactas", "clave_repetida", "calificacion_fuera_norma"]):
        st.caption(f"Depuracion aplicada: {informe}")

    st.write("")
    conf = u.config_columnas_p()
    conf["votos_anomalia"] = st.column_config.NumberColumn("Anomalia", format="%d / 3")
    for n in ["K-Means", "DBSCAN", "Isolation F."]:
        conf[n] = st.column_config.TextColumn(n)
    res = res.copy()
    res["K-Means"], res["DBSCAN"], res["Isolation F."] = (
        res.flag_kmeans.apply(u.marca), res.flag_dbscan.apply(u.marca), res.flag_if.apply(u.marca))
    base_cols = ["NUM_OPER", "P_xgboost", "P_lightgbm", "P_ensemble", "votos_anomalia",
                 "K-Means", "DBSCAN", "Isolation F.", "DIASMORA", "particion"]

    t1, t2, t3 = st.tabs(["Operaciones nuevas", "Anomalias por modelo en este corte", "Todo el corte"])
    with t1:
        if nuevas.empty:
            st.info("El archivo no trae operaciones nuevas.")
        else:
            st.dataframe(res[res.particion == "nuevo"][base_cols].sort_values(P, ascending=False),
                         column_config=conf, hide_index=True, width="stretch")
            dif = res[(res.particion == "nuevo") & ((res.P_xgboost - res.P_lightgbm).abs() > 0.3)]
            if len(dif):
                st.warning(f"Los modelos discrepan fuertemente (diferencia > 0.30) en: "
                           f"{', '.join(map(str, dif.NUM_OPER))}. Conviene revision manual.")
    with t2:
        col_k, col_d, col_i = st.columns(3)
        for col, nombre, flag, score, asc in [
            (col_k, "K-Means", "flag_kmeans", "dist_cluster_riesgo", True),
            (col_d, "DBSCAN", "flag_dbscan", "dist_nucleo_dbscan", False),
            (col_i, "Isolation Forest", "flag_if", "score_if", False)]:
            sel = res[res[flag] == 1].sort_values(score, ascending=asc)
            col.markdown(f"**{nombre}** · {len(sel)} senaladas")
            col.dataframe(sel[["NUM_OPER", P, "DIASMORA"]], column_config=conf,
                          hide_index=True, width="stretch", height=320)
    with t3:
        st.dataframe(res.sort_values("score_anomalia", ascending=False)[base_cols],
                     column_config=conf, hide_index=True, width="stretch")
        u.tabla_descarga(res[base_cols], f"vista_previa_{corte:%Y_%m}.csv")

    # ------------------------------------------------------------------ Incorporar
    st.write("")
    with st.container(border=True):
        st.markdown("**Incorporar al historico**")
        st.caption("Guarda el archivo en `data/raw/cortes_nuevos/` (el CSV original no se modifica) "
                   "y recalcula todas las paginas: cartera vigente, trayectorias, anomalias y SHAP.")
        if st.button(f"Incorporar corte {corte:%d/%m/%Y}", type="primary"):
            with st.spinner("Incorporando y recalculando el scoring de toda la cartera..."):
                ruta = sc.incorporar_corte(df, config)
            u.refrescar_datos()
            st.success(f"Corte incorporado ({os.path.basename(ruta)}). Todas las paginas ya muestran "
                       f"el corte {corte:%d/%m/%Y}.")

# ------------------------------------------------------------------ Historial de cortes
st.write("")
with st.expander("Cortes incorporados despues del entrenamiento"):
    carpeta = sc.carpeta_cortes_nuevos(config)
    archivos = sorted(glob.glob(os.path.join(carpeta, "*.csv")))
    if not archivos:
        st.write("Ninguno todavia. El tablero muestra el ultimo corte del entrenamiento.")
    else:
        st.write(", ".join(os.path.basename(a) for a in archivos))
        confirmar = st.checkbox(f"Confirmo que deseo retirar {os.path.basename(archivos[-1])}")
        if st.button("Deshacer ultimo corte", disabled=not confirmar):
            os.remove(archivos[-1])
            with st.spinner("Recalculando..."):
                sc.run(config, silencioso=True)
            u.refrescar_datos()
            st.success("Ultimo corte retirado y tablero recalculado.")
