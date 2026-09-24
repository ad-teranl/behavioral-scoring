"""
cierre_integrador.py
====================
Cruza los resultados de los 3 modelos no supervisados:
  - Identifica operaciones senaladas por 2 o mas modelos como anomalas
  - Aplica XGBoost para asignar P(mora > 30 dias) a cada anomalia
  - Genera graficos SHAP (beeswarm global + waterfall por operacion)

Esta es la pieza que convierte el hallazgo no supervisado en una
estimacion de riesgo accionable para el comite de cobranza.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def run(datos_sup, datos_nosup, modelos_supervisados, modelos_nosupervisados, config):
    """
    Cierre integrador: anomalias en 2+ modelos -> XGBoost -> P(mora) + SHAP.

    Parametros:
        datos_sup          : dict con X_te_sup, y_te, X_tr_sup, y_tr
        datos_nosup        : dict con X_te_nosup, y_te
        modelos_supervisados  : dict con 'xgboost', 'lightgbm'
        modelos_nosupervisados: dict con 'idx_km', 'idx_db', 'idx_if'
    """
    ruta_reports = config.paths.reports
    X_te_sup     = datos_sup['X_te_sup']
    X_te_nosup   = datos_nosup['X_te_nosup']
    y_te         = datos_nosup['y_te']
    modelo_xgb   = modelos_supervisados['xgboost']

    idx_km = modelos_nosupervisados['idx_km']
    idx_db = modelos_nosupervisados['idx_db']
    idx_if = modelos_nosupervisados['idx_if']

    # ----------------------------------------------------------------
    # 1. OPERACIONES SENALADAS POR 2+ MODELOS
    # ----------------------------------------------------------------
    print("\n" + "="*55)
    print("CIERRE INTEGRADOR")
    print("="*55)

    total_test = len(X_te_nosup)
    votos = np.zeros(total_test, dtype=int)
    votos[idx_km] += 1
    votos[idx_db] += 1
    votos[idx_if] += 1

    mask_anomalos = votos >= 2
    n_anomalos    = mask_anomalos.sum()
    print(f"\n  Operaciones en test:           {total_test}")
    print(f"  Anomalas en 2+ modelos:        {n_anomalos} ({n_anomalos/total_test*100:.1f}%)")

    if n_anomalos == 0:
        print("  Sin anomalias detectadas por 2+ modelos. Ajusta eps, contamination o n_clusters.")
        return {}

    # ----------------------------------------------------------------
    # 2. SUBSET DE ANOMALOS — variables supervisadas para XGBoost
    # ----------------------------------------------------------------
    # Alineamos por posicion: X_te_sup y X_te_nosup tienen el mismo
    # numero de filas (mismo conjunto de prueba, distinto set de columnas)
    X_anomalos = X_te_sup.iloc[mask_anomalos]
    y_anomalos = y_te.iloc[mask_anomalos]

    # ----------------------------------------------------------------
    # 3. PROBABILIDADES DE MORA CON XGBOOST
    # ----------------------------------------------------------------
    proba_mora = modelo_xgb.predict_proba(X_anomalos)[:, 1]
    y_pred     = modelo_xgb.predict(X_anomalos)

    # Tabla de resultados
    df_result = X_anomalos.copy()
    df_result['TARGET_real']  = y_anomalos.values
    df_result['P_mora_xgb']   = proba_mora
    df_result['Pred_mora_xgb']= y_pred
    df_result['votos_nosup']  = votos[mask_anomalos]

    # Zona gris (0.40 < P < 0.60)
    df_result['zona_gris'] = df_result['P_mora_xgb'].between(0.40, 0.60)

    print(f"\n  Promedio P(mora) en anomalos:  {proba_mora.mean():.4f}")
    print(f"  Anomalos con mora real:        {int(y_anomalos.sum())} / {n_anomalos}")
    print(f"  En zona gris (0.40-0.60):      {df_result['zona_gris'].sum()}")

    # Distribucion por umbral
    for umbral in [0.30, 0.45, 0.50, 0.60]:
        n = (proba_mora >= umbral).sum()
        print(f"  P(mora) >= {umbral}: {n} operaciones")

    # Exportar tabla
    df_result.to_csv(f"{ruta_reports}anomalos_con_probabilidad.csv", index=True)
    print(f"\n  Tabla exportada: {ruta_reports}anomalos_con_probabilidad.csv")

    # ----------------------------------------------------------------
    # 4. SHAP — BEESWARM GLOBAL
    # ----------------------------------------------------------------
    print("\n  Calculando SHAP values...")
    explainer   = shap.TreeExplainer(modelo_xgb)
    shap_values = explainer.shap_values(X_anomalos)

    plt.figure()
    shap.summary_plot(shap_values, X_anomalos, plot_type="dot", show=False)
    plt.title("SHAP — XGBoost sobre anomalos (2+ modelos)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{ruta_reports}shap_beeswarm_anomalos.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SHAP beeswarm guardado: {ruta_reports}shap_beeswarm_anomalos.png")

    # ----------------------------------------------------------------
    # 5. SHAP WATERFALL — top 5 anomalos por P(mora)
    # ----------------------------------------------------------------
    top5_idx = np.argsort(proba_mora)[::-1][:5]

    for rank, i in enumerate(top5_idx):
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[i],
                base_values=explainer.expected_value,
                data=X_anomalos.iloc[i],
                feature_names=list(X_anomalos.columns)
            ),
            show=False
        )
        num_oper = X_anomalos.index[i]
        p_mora   = proba_mora[i]
        plt.title(f"Waterfall #{rank+1} — NUM_OPER: {num_oper} | P(mora)={p_mora:.4f}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{ruta_reports}shap_waterfall_top{rank+1}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  SHAP waterfall (top 5) guardados en: {ruta_reports}")
    print("\n  Cierre integrador completado.")

    return {
        'df_anomalos': df_result,
        'proba_mora':  proba_mora,
        'shap_values': shap_values,
    }
