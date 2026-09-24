"""
Anomalias por modelo — lo que detecta CADA detector por separado,
ordenado por su propio puntaje, y cuanto coincide con los otros dos.
Complementa (no reemplaza) el consenso de la pagina de Inicio.
"""
import plotly.express as px
import streamlit as st

from app import utils as u

st.set_page_config(page_title="Behavioral Scoring | Anomalias por modelo", page_icon="🧭", layout="wide")
datos = u.datos_o_detener()
modelo, umbral = u.barra_lateral(datos)
vig = datos["cartera_vigente"].copy()
P = u.col_p(modelo)

u.aplicar_estilo("Anomalias por modelo",
                 "Que operaciones senala cada detector por si solo: K-Means, DBSCAN e Isolation Forest")

DETECTORES = {
    "K-Means": dict(flag="flag_kmeans", score="dist_cluster_riesgo", asc=True,
                    etiqueta="Distancia al cluster de riesgo",
                    ayuda="Menor distancia = mas parecida al grupo de mayor mora. "
                          "Senala a las operaciones asignadas a ese cluster."),
    "DBSCAN": dict(flag="flag_dbscan", score="dist_nucleo_dbscan", asc=False,
                   etiqueta="Distancia al nucleo mas cercano",
                   ayuda="Mayor distancia = mas aislada de cualquier grupo denso. "
                         "Senala como ruido a las que superan el radio eps."),
    "Isolation Forest": dict(flag="flag_if", score="score_if", asc=False,
                             etiqueta="Puntaje de aislamiento",
                             ayuda="Mayor puntaje = mas facil de aislar = mas atipica. "
                                   "Senala al porcentaje definido en contamination."),
}

# ------------------------------------------------------------------ KPIs por detector
cols = st.columns(3)
for col, (nombre, d) in zip(cols, DETECTORES.items()):
    flag = vig[d["flag"]] == 1
    otros = [x["flag"] for n, x in DETECTORES.items() if n != nombre]
    exclusivas = int((flag & (vig[otros].sum(axis=1) == 0)).sum())
    u.kpi(col, nombre, f"{int(flag.sum()):,}",
          f"{exclusivas} exclusivas · {int((flag & (vig.en_mora == 1)).sum())} ya en mora", u.AMBAR)
st.write("")

# ------------------------------------------------------------------ Controles
with st.container(border=True):
    a, b, c, d_ = st.columns([1.6, 1, 1.1, 1.1])
    elegido = a.radio("Detector", list(DETECTORES), horizontal=True)
    top_n = b.number_input("Cantidad a mostrar", min_value=1, max_value=len(vig), value=20, step=5)
    excluir = c.checkbox("Excluir ya en mora", value=True)
    solo_excl = d_.checkbox("Solo exclusivas de este modelo",
                            help="Operaciones que solo este detector senala.")

cfg = DETECTORES[elegido]
st.caption(cfg["ayuda"])

vista = vig[vig[cfg["flag"]] == 1].copy()
if excluir:
    vista = vista[vista.en_mora == 0]
if solo_excl:
    otros = [x["flag"] for n, x in DETECTORES.items() if n != elegido]
    vista = vista[vista[otros].sum(axis=1) == 0]
vista = vista.sort_values(cfg["score"], ascending=cfg["asc"]).head(int(top_n))

if vista.empty:
    st.info(f"{elegido} no senala operaciones con estos filtros. "
            "Prueba incluyendo las que ya estan en mora.")
else:
    vista.insert(0, "Prioridad", range(1, len(vista) + 1))
    for nombre, x in DETECTORES.items():
        vista[nombre] = vista[x["flag"]].apply(u.marca)
    columnas = ["Prioridad", "NUM_OPER", cfg["score"], "K-Means", "DBSCAN", "Isolation Forest",
                "votos_anomalia", "P_xgboost", "P_lightgbm", "DIASMORA", "particion"]
    conf = u.config_columnas_p()
    conf.update({
        cfg["score"]: st.column_config.NumberColumn(cfg["etiqueta"], format="%.3f"),
        "votos_anomalia": st.column_config.NumberColumn("Total modelos", format="%d / 3"),
    })
    st.dataframe(vista[columnas], column_config=conf, hide_index=True, width="stretch")
    confirm = int((vista[P] >= umbral).sum())
    st.caption(f"{confirm} de {len(vista)} superan el umbral {umbral:.2f} con {u.ETIQUETA_MODELO[modelo]}.")
    u.tabla_descarga(vista[columnas], f"anomalias_{elegido.replace(' ', '_').lower()}.csv")

# ------------------------------------------------------------------ Coincidencias
st.write("")
combo = vig.copy()
combo["Combinacion"] = combo.apply(
    lambda r: " + ".join(n for n, x in DETECTORES.items() if r[x["flag"]] == 1) or "Ninguno", axis=1)
resumen = combo[combo.Combinacion != "Ninguno"].groupby(["Combinacion", "en_mora"]).size().reset_index(name="operaciones")
resumen["Estado"] = resumen.en_mora.map({0: "Sin mora", 1: "En mora"})
fig = px.bar(resumen, y="Combinacion", x="operaciones", color="Estado", orientation="h",
             color_discrete_map={"Sin mora": u.VERDE, "En mora": u.ROJO},
             title="Coincidencias entre detectores (solo operaciones senaladas)")
fig.update_layout(plot_bgcolor="white", height=380, legend_title=None, yaxis_title=None,
                  xaxis_title="Operaciones")
st.plotly_chart(fig, width="stretch")
