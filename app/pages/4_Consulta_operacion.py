"""
Consulta de operacion — ingresa un NUM_OPER y obtiene:
P(mora) por modelo, semaforo, trayectoria mes a mes y explicacion SHAP.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import utils as u

st.set_page_config(page_title="Behavioral Scoring | Consulta", page_icon="🔎", layout="wide")
datos = u.datos_o_detener()
modelo, umbral = u.barra_lateral(datos)
umbrales = u.cargar_config()["dashboard"]["umbrales"]
vig = datos["cartera_vigente"]
hist = datos["historial_puntuado"]
P = u.col_p(modelo)

u.aplicar_estilo("Consulta de operacion",
                 "Paso 4 · Probabilidad de incumplimiento, trayectoria y factores que la explican")

ops = sorted(vig.NUM_OPER.astype(int).tolist())
with st.container(border=True):
    a, b, c = st.columns([1.2, 1, 1])
    num = a.number_input("Numero de operacion", min_value=int(min(ops)), max_value=int(max(ops)),
                         value=int(st.session_state.get("num_oper", ops[0])), step=1)
    modelo_shap = b.selectbox("Explicar con", ["xgboost", "lightgbm"], format_func=u.ETIQUETA_MODELO.get,
                              index=0 if modelo != "lightgbm" else 1)
    c.write("")
    consultar = c.button("Consultar", type="primary", width="stretch")

if consultar:
    st.session_state.num_oper = int(num)
num = st.session_state.get("num_oper", int(num))

if num not in ops:
    st.warning(f"La operacion {num} no esta en la cartera vigente del ultimo corte "
               "(pudo cancelarse, castigarse o no existir).")
    st.stop()

fila = vig[vig.NUM_OPER == num].iloc[0]
p = float(fila[P])
nivel = u.nivel_riesgo(p, umbrales)

# ------------------------------------------------------------------ Tarjetas
c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1])
c1.markdown(f"""<div class="semaforo" style="background:{u.COLORES_NIVEL[nivel]}">
    <div style="font-size:.85rem;opacity:.9">Operacion {num} · {u.ETIQUETA_MODELO[modelo]}</div>
    <div class="n">P(mora) = {p:.3f}</div><div>{u.NIVELES[nivel]}</div></div>""",
            unsafe_allow_html=True)
u.kpi(c2, "XGBoost", f"{fila.P_xgboost:.3f}")
u.kpi(c3, "LightGBM", f"{fila.P_lightgbm:.3f}")
dist = umbral - p
u.kpi(c4, f"Distancia a {umbral:.2f}", f"{dist:+.3f}", "negativa = ya lo supera",
      u.ROJO if dist <= 0 else u.VERDE)
u.kpi(c5, "Dias de mora", f"{int(fila.DIASMORA)}", f"anomalia {int(fila.votos_anomalia)}/3 modelos",
      u.ROJO if fila.en_mora else u.AZUL)

if fila.particion == "nuevo":
    st.caption("Operacion nueva, posterior al entrenamiento: los modelos nunca la vieron "
               "y aun no tiene historia suficiente para calcular tendencia.")
elif fila.particion == "train":
    st.caption("Nota: esta operacion pertenece al conjunto de entrenamiento; el modelo ya la conocia, "
               "por lo que su probabilidad puede ser mas extrema que en datos no vistos.")

# ------------------------------------------------------------------ Trayectoria
g1, g2 = st.columns([1.1, 1])
with g1:
    h = hist[hist.NUM_OPER == num].sort_values("FECHA_CORTE")
    fig = go.Figure()
    for col, nombre, color in [("P_xgboost", "XGBoost", u.AZUL), ("P_lightgbm", "LightGBM", u.VERDE),
                               ("P_ensemble", "Ensemble", u.GRIS)]:
        fig.add_trace(go.Scatter(x=h.FECHA_CORTE, y=h[col], mode="lines+markers", name=nombre,
                                 line=dict(color=color, width=3 if col == P else 1.5)))
    fig.add_hline(y=umbral, line_dash="dash", line_color=u.ROJO, annotation_text=f"Umbral {umbral:.2f}")
    fig.update_layout(title=f"Trayectoria de P(mora) · {len(h)} cortes", plot_bgcolor="white",
                      height=380, yaxis=dict(range=[0, 1], title="P(mora)"), xaxis_title="Corte",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    if len(h) < 2:
        st.caption("Solo tiene un corte: no hay trayectoria para calcular tendencia.")
    else:
        st.caption(f"Pendiente reciente: {fila.pendiente_mensual*100:+.2f} puntos porcentuales por mes.")

# ------------------------------------------------------------------ SHAP waterfall
with g2:
    shp = datos[f"shap_{modelo_shap}"]
    base = float(datos["shap_base"].set_index("modelo").loc[modelo_shap, "base_logodds"])
    valores = shp[shp.NUM_OPER == num].drop(columns="NUM_OPER").iloc[0]
    orden = valores.abs().sort_values(ascending=True).index
    valores = valores[orden]
    nombres = [u.NOMBRES.get(v, v) for v in valores.index]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=["Valor base"] + nombres + ["Prediccion"],
        x=[base] + valores.tolist() + [0],
        measure=["absolute"] + ["relative"] * len(valores) + ["total"],
        increasing=dict(marker=dict(color=u.ROJO)), decreasing=dict(marker=dict(color=u.VERDE)),
        totals=dict(marker=dict(color=u.AZUL_OSCURO)),
        texttemplate="%{delta:+.2f}", textposition="outside"))
    fig.update_layout(title=f"Factores que explican la probabilidad · SHAP {u.ETIQUETA_MODELO[modelo_shap]}",
                      plot_bgcolor="white", height=380, xaxis_title="Contribucion (log-odds)",
                      margin=dict(l=10, r=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("Rojo: empuja hacia el incumplimiento · Verde: lo reduce. "
               "Escala log-odds; la probabilidad es la transformacion logistica del total.")
