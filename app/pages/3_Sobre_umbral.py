"""
Sobre el umbral — operaciones que el modelo ya clasifica como incumplimiento.
Pestaña 1: aun al dia pero con P(mora) >= umbral (intervencion inmediata).
Pestaña 2: ya en mora (>30 dias) — gestion de cobranza/recuperacion.
"""
import plotly.express as px
import streamlit as st

import utils as u

st.set_page_config(page_title="Behavioral Scoring | Sobre umbral", page_icon="🚨", layout="wide")
datos = u.datos_o_detener()
modelo, umbral = u.barra_lateral(datos)
umbrales = u.cargar_config()["dashboard"]["umbrales"]
vig = datos["cartera_vigente"].copy()
P = u.col_p(modelo)

u.aplicar_estilo("Operaciones sobre el umbral",
                 "Paso 3 · Clasificadas como incumplimiento por el modelo: intervencion y cobranza")

vig["nivel"] = vig[P].apply(lambda p: u.nivel_riesgo(p, umbrales))
vig["Nivel de riesgo"] = vig.nivel.map(dict(enumerate(u.NIVELES)))

sobre_sin_mora = vig[(vig.en_mora == 0) & (vig[P] >= umbral)].sort_values(P, ascending=False)
en_mora = vig[vig.en_mora == 1].sort_values(P, ascending=False)

c1, c2, c3 = st.columns(3)
u.kpi(c1, "Al dia pero sobre umbral", f"{len(sobre_sin_mora):,}", "intervencion inmediata", u.NARANJA)
u.kpi(c2, "Ya en mora", f"{len(en_mora):,}", "gestion de recuperacion", u.ROJO)
u.kpi(c3, "En mora y sobre umbral", f"{int((en_mora[P] >= umbral).sum()):,}",
      f"de {len(en_mora)} · coherencia del modelo", u.AZUL)
st.write("")

columnas = ["NUM_OPER", "Nivel de riesgo", "P_xgboost", "P_lightgbm", "P_ensemble",
            "DIASMORA", "votos_anomalia", "n_cortes", "particion"]
conf = u.config_columnas_p()
conf.update({"votos_anomalia": st.column_config.NumberColumn("Anomalia", format="%d / 3"),
             "n_cortes": st.column_config.NumberColumn("Cortes", format="%d"),
             "particion": st.column_config.TextColumn("Particion")})

t1, t2 = st.tabs([f"Al dia pero sobre umbral ({len(sobre_sin_mora)})", f"Ya en mora ({len(en_mora)})"])
with t1:
    if sobre_sin_mora.empty:
        st.success(f"Ninguna operacion al dia supera el umbral {umbral:.2f} con {u.ETIQUETA_MODELO[modelo]}.")
    else:
        st.dataframe(sobre_sin_mora[columnas], column_config=conf, hide_index=True, width="stretch")
        u.tabla_descarga(sobre_sin_mora[columnas], f"al_dia_sobre_umbral_{umbral:.2f}.csv")
with t2:
    st.dataframe(en_mora[columnas], column_config=conf, hide_index=True, width="stretch")
    u.tabla_descarga(en_mora[columnas], "operaciones_en_mora.csv")

st.write("")
dist = vig.groupby(["nivel", "Nivel de riesgo"]).size().reset_index(name="operaciones").sort_values("nivel")
fig = px.bar(dist, x="Nivel de riesgo", y="operaciones", color="Nivel de riesgo",
             color_discrete_sequence=[u.COLORES_NIVEL[i] for i in dist.nivel],
             title=f"Cartera vigente por nivel de riesgo (umbrales {', '.join(f'{x:.2f}' for x in umbrales)})")
fig.update_layout(plot_bgcolor="white", height=340, showlegend=False, yaxis_title="Operaciones")
st.plotly_chart(fig, width="stretch")
