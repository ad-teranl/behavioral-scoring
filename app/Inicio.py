"""
Inicio — Resumen de cartera vigente y deteccion de anomalias.
Las anomalias (3 modelos no supervisados) son los primeros indicios de riesgo;
luego se contrastan con la P(mora) de los modelos supervisados.
"""
import plotly.express as px
import streamlit as st

import utils as u

st.set_page_config(page_title="Behavioral Scoring | Inicio", page_icon="📊", layout="wide")
datos = u.datos_o_detener()
modelo, umbral = u.barra_lateral(datos)
cfg = u.cargar_config()["dashboard"]
vig = datos["cartera_vigente"].copy()
P = u.col_p(modelo)

u.aplicar_estilo("Monitoreo de cartera vigente",
                 "Paso 1 · Anomalias detectadas por K-Means, DBSCAN e Isolation Forest, "
                 "contrastadas con la probabilidad de incumplimiento")

# ------------------------------------------------------------------ KPIs
c1, c2, c3, c4, c5 = st.columns(5)
n_nuevas = int((vig.particion == "nuevo").sum())
u.kpi(c1, "Operaciones vigentes", f"{len(vig):,}",
      f"{n_nuevas} nuevas desde el entrenamiento" if n_nuevas else "presentes en el ultimo corte")
u.kpi(c2, "Ya en mora", f"{int(vig.en_mora.sum()):,}",
      f"{vig.en_mora.mean()*100:.1f}% · mas de 30 dias", u.ROJO)
u.kpi(c3, "Sin mora", f"{int((vig.en_mora == 0).sum()):,}", "universo preventivo", u.VERDE)
u.kpi(c4, "Anomalas (2+ modelos)", f"{int((vig.votos_anomalia >= 2).sum()):,}",
      "consenso de detectores", u.AMBAR)
u.kpi(c5, f"Sobre umbral {umbral:.2f}", f"{int((vig[P] >= umbral).sum()):,}",
      f"segun {u.ETIQUETA_MODELO[modelo]}", u.NARANJA)

st.write("")

# ------------------------------------------------------------------ Controles
st.subheader("Operaciones que requieren investigacion")
with st.container(border=True):
    a, b, c, d = st.columns([1, 1, 1.3, 1])
    top_n = a.number_input("Cantidad a investigar", min_value=1, max_value=len(vig),
                           value=int(cfg["top_n_anomalos"]), step=5)
    min_votos = b.number_input("Minimo de modelos que la senalan", min_value=0, max_value=3,
                               value=1, step=1)
    excluir = c.checkbox("Excluir operaciones ya en mora", value=bool(cfg["excluir_en_mora"]),
                         help="Activado: muestra anomalias que aun se pueden prevenir.")
    d.write("")
    generar = d.button("Generar lista", type="primary", width="stretch")

if generar or "lista_anom" not in st.session_state:
    base = vig[vig.votos_anomalia >= min_votos]
    if excluir:
        base = base[base.en_mora == 0]
    st.session_state.lista_anom = base.sort_values("score_anomalia", ascending=False).head(int(top_n))

lista = st.session_state.lista_anom.copy()

if lista.empty:
    st.warning("Ninguna operacion cumple los criterios. Reduce el minimo de modelos o incluye las que ya estan en mora.")
else:
    lista["Confirmacion supervisada"] = lista[P].apply(
        lambda p: "Confirma riesgo" if p >= umbral else "No confirma")
    for col, nombre in [("flag_kmeans", "K-Means"), ("flag_dbscan", "DBSCAN"), ("flag_if", "Isolation F.")]:
        lista[nombre] = lista[col].map({1: "✔", 0: "—"})
    lista.insert(0, "Prioridad", range(1, len(lista) + 1))

    columnas = ["Prioridad", "NUM_OPER", "votos_anomalia", "K-Means", "DBSCAN", "Isolation F.",
                "score_anomalia", "P_xgboost", "P_lightgbm", "P_ensemble", "DIASMORA",
                "Confirmacion supervisada", "particion"]
    conf = u.config_columnas_p()
    conf.update({
        "votos_anomalia": st.column_config.NumberColumn("Modelos que la senalan", format="%d / 3"),
        "score_anomalia": st.column_config.NumberColumn("Puntaje anomalia", format="%.2f",
                                                        help="Votos (0-3) + intensidad promedio (0-1)"),
        "particion": st.column_config.TextColumn("Particion", help="train = el modelo supervisado ya la vio"),
    })
    st.dataframe(lista[columnas], column_config=conf, hide_index=True, width="stretch")

    confirmadas = int((lista[P] >= umbral).sum())
    st.caption(f"{confirmadas} de {len(lista)} operaciones listadas superan el umbral {umbral:.2f} "
               f"con {u.ETIQUETA_MODELO[modelo]}. Consulta el detalle de cualquiera en la pagina "
               f"**Consulta operacion**.")
    u.tabla_descarga(lista[columnas], "anomalias_priorizadas.csv")

# ------------------------------------------------------------------ Graficos
st.write("")
g1, g2 = st.columns(2)
with g1:
    conteo = vig.groupby(["votos_anomalia", "en_mora"]).size().reset_index(name="operaciones")
    conteo["Estado"] = conteo.en_mora.map({0: "Sin mora", 1: "En mora"})
    fig = px.bar(conteo, x="votos_anomalia", y="operaciones", color="Estado", barmode="group",
                 color_discrete_map={"Sin mora": u.VERDE, "En mora": u.ROJO},
                 labels={"votos_anomalia": "Modelos que senalan la operacion", "operaciones": "Operaciones"},
                 title="Consenso de detectores de anomalias")
    fig.update_layout(plot_bgcolor="white", height=360, legend_title=None)
    st.plotly_chart(fig, width="stretch")
with g2:
    vg = vig.copy()
    vg["Estado"] = vg.en_mora.map({0: "Sin mora", 1: "En mora"})
    fig = px.scatter(vg, x="score_anomalia", y=P, color="Estado", hover_data=["NUM_OPER", "DIASMORA"],
                     color_discrete_map={"Sin mora": u.VERDE, "En mora": u.ROJO},
                     labels={"score_anomalia": "Puntaje de anomalia", P: "P(mora)"},
                     title="Anomalia vs probabilidad de incumplimiento")
    fig.add_hline(y=umbral, line_dash="dash", line_color=u.AZUL_OSCURO,
                  annotation_text=f"Umbral {umbral:.2f}")
    fig.update_layout(plot_bgcolor="white", height=360, legend_title=None)
    st.plotly_chart(fig, width="stretch")
