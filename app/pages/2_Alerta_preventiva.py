"""
Alerta preventiva — operaciones SIN mora que aun no cruzan el umbral,
priorizadas por cercania al umbral o por velocidad de acercamiento.
"""
import numpy as np
import plotly.express as px
import streamlit as st

from app import utils as u

st.set_page_config(page_title="Behavioral Scoring | Alerta preventiva", page_icon="⚠️", layout="wide")
datos = u.datos_o_detener()
modelo, umbral = u.barra_lateral(datos)
vig = datos["cartera_vigente"].copy()
P = u.col_p(modelo)

u.aplicar_estilo("Alerta preventiva",
                 "Paso 2 · Socios al dia que se acercan al umbral de incumplimiento: actuar antes de la mora")

# Universo: sin mora y aun por debajo del umbral
base = vig[(vig.en_mora == 0) & (vig[P] < umbral)].copy()
base["distancia"] = umbral - base[P]
base["meses_al_umbral"] = np.where(base.pendiente_mensual > 0,
                                   base.distancia / base.pendiente_mensual, np.inf)
base["Tendencia"] = np.select(
    [base.n_cortes < 2, base.pendiente_mensual > 1e-4, base.pendiente_mensual < -1e-4],
    ["Sin historia", "Empeorando", "Mejorando"], default="Estable")

# ------------------------------------------------------------------ KPIs
c1, c2, c3, c4 = st.columns(4)
u.kpi(c1, "En vigilancia", f"{len(base):,}", f"sin mora y bajo {umbral:.2f}")
u.kpi(c2, "Empeorando", f"{int((base.Tendencia == 'Empeorando').sum()):,}",
      "P(mora) en aumento", u.NARANJA)
u.kpi(c3, "Cruzarian en ≤ 6 meses", f"{int((base.meses_al_umbral <= 6).sum()):,}",
      "proyeccion lineal", u.ROJO)
cerca = base.distancia.min() if len(base) else np.nan
u.kpi(c4, "Distancia minima", f"{cerca:.3f}" if len(base) else "—", "la mas cercana al umbral", u.AMBAR)

st.write("")

# ------------------------------------------------------------------ Controles
with st.container(border=True):
    a, b, c, d = st.columns([1.4, 1, 1, 1])
    criterio = a.radio("Priorizar por", ["Menor distancia al umbral", "Menos meses al umbral"],
                       horizontal=True)
    top_n = b.number_input("Operaciones a mostrar", min_value=1, max_value=max(len(base), 1),
                           value=min(25, max(len(base), 1)), step=5)
    solo_hist = c.checkbox("Solo con trayectoria (2+ cortes)")
    solo_emp = d.checkbox("Solo empeorando")

vista = base.copy()
if solo_hist:
    vista = vista[vista.n_cortes >= 2]
if solo_emp:
    vista = vista[vista.Tendencia == "Empeorando"]
orden = "distancia" if criterio.startswith("Menor distancia") else "meses_al_umbral"
vista = vista.sort_values([orden, "distancia"]).head(int(top_n))

if vista.empty:
    st.warning("Ninguna operacion cumple los filtros seleccionados.")
else:
    vista.insert(0, "Prioridad", range(1, len(vista) + 1))
    vista["pendiente_pp"] = vista.pendiente_mensual * 100
    vista["meses_txt"] = vista.meses_al_umbral.apply(lambda m: "—" if np.isinf(m) else f"{m:.1f}")
    columnas = ["Prioridad", "NUM_OPER", P, "distancia", "Tendencia", "pendiente_pp", "meses_txt",
                "n_cortes", "votos_anomalia", "DIASMORA", "particion"]
    conf = u.config_columnas_p()
    conf.update({
        "distancia": st.column_config.NumberColumn("Distancia al umbral", format="%.3f"),
        "pendiente_pp": st.column_config.NumberColumn("Pendiente (pp/mes)", format="%+.2f",
                                                      help="Puntos porcentuales de P(mora) por mes"),
        "meses_txt": st.column_config.TextColumn("Meses al umbral"),
        "n_cortes": st.column_config.NumberColumn("Cortes de historia", format="%d"),
        "votos_anomalia": st.column_config.NumberColumn("Anomalia", format="%d / 3"),
        "particion": st.column_config.TextColumn("Particion"),
    })
    st.dataframe(vista[columnas], column_config=conf, hide_index=True, width="stretch")
    u.tabla_descarga(vista[columnas], f"alerta_preventiva_umbral_{umbral:.2f}.csv")

# ------------------------------------------------------------------ Distribucion
st.write("")
fig = px.histogram(vig[vig.en_mora == 0], x=P, nbins=50, color_discrete_sequence=[u.AZUL],
                   labels={P: "P(mora)"}, title="Distribucion de P(mora) en operaciones sin mora")
fig.add_vline(x=umbral, line_dash="dash", line_color=u.ROJO, annotation_text=f"Umbral {umbral:.2f}")
fig.update_layout(plot_bgcolor="white", height=330, yaxis_title="Operaciones")
st.plotly_chart(fig, width="stretch")

if len(base) and cerca > 0.20:
    st.info(f"La operacion sin mora mas cercana esta a {cerca:.2f} del umbral. El modelo separa con mucha "
            "claridad a los socios al dia de los morosos, por eso las probabilidades de la cartera sana "
            "son bajas. El orden relativo y la tendencia siguen siendo la senal util para priorizar.")
st.caption("Meses al umbral = distancia ÷ pendiente mensual (ultimos cortes). Es una proyeccion lineal "
           "orientativa, no una prediccion exacta de la fecha de incumplimiento.")
