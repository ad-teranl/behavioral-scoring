"""
train_supervised.py
===================
Entrena XGBoost y LightGBM con flags de reentrenamiento independientes.

Control desde main.yaml:
  reentrenar_xgboost:  false/true
  reentrenar_lightgbm: false/true
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
from scipy.stats import ks_2samp


def calcular_metricas(nombre, modelo, X_test, y_test, ruta_reports):
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    mcc   = matthews_corrcoef(y_test, y_pred)

    proba_pos  = y_proba[y_test == 1]
    proba_neg  = y_proba[y_test == 0]
    ks_stat, _ = ks_2samp(proba_pos, proba_neg)

    print(f"\n{'='*55}")
    print(f"  Metricas: {nombre}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc:.4f}   Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}   F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}   KS Stat   : {ks_stat:.4f}")
    print(f"  Brier     : {brier:.4f}   MCC       : {mcc:.4f}")

    # Matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title(f'Confusion Matrix — {nombre}')
    plt.tight_layout()
    nombre_limpio = nombre.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f"{ruta_reports}cm_{nombre_limpio}.png", dpi=150)
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC — {nombre}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ruta_reports}roc_{nombre_limpio}.png", dpi=150)
    plt.close()

    return {
        'Modelo': nombre, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1': f1, 'AUC': auc, 'KS': ks_stat,
        'Brier': brier, 'MCC': mcc
    }


def train_xgboost(datos_sup, config, ruta_models, ruta_reports, reentrenar):
    X_tr  = datos_sup['X_tr_sup']
    X_val = datos_sup['X_val_sup']
    X_te  = datos_sup['X_te_sup']
    y_tr  = datos_sup['y_tr']
    y_val = datos_sup['y_val']
    y_te  = datos_sup['y_te']

    ruta_gs    = f"{ruta_models}gs_XGBoost.pkl"
    ruta_model = f"{ruta_models}modelo_XGBoost.pkl"
    cfg = config.xgboost

    if reentrenar or not (os.path.exists(ruta_gs) and os.path.exists(ruta_model)):
        print("\n[XGBoost] Iniciando GridSearchCV...")
        param_grid = {
            'n_estimators':     list(cfg.n_estimators_grid),
            'max_depth':        list(cfg.max_depth_grid),
            'learning_rate':    list(cfg.learning_rate_grid),
            'subsample':        list(cfg.subsample_grid),
            'colsample_bytree': list(cfg.colsample_bytree_grid),
            'min_child_weight': list(cfg.min_child_weight_grid),
            'gamma':            list(cfg.gamma_grid),
        }
        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        gs = GridSearchCV(
            estimator=XGBClassifier(random_state=cfg.random_state, eval_metric='auc', n_jobs=-1),
            param_grid=param_grid, scoring=cfg.scoring, cv=cv, n_jobs=-1, verbose=1
        )
        gs.fit(X_tr, y_tr)
        joblib.dump(gs, ruta_gs)

        best_params = gs.best_params_.copy()
        best_params.pop('n_estimators', None)
        print(f"\n[XGBoost] Mejores parametros: {gs.best_params_}")
        print(f"[XGBoost] Mejor F1 CV: {gs.best_score_:.4f}")

        modelo = XGBClassifier(
            **best_params,
            n_estimators=cfg.n_estimators_final,
            early_stopping_rounds=cfg.early_stopping_rounds,
            eval_metric=cfg.eval_metric,
            random_state=cfg.random_state,
            n_jobs=-1
        )
        modelo.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=50)
        joblib.dump(modelo, ruta_model)
        print(f"[XGBoost] Arboles optimos: {modelo.best_iteration}")
    else:
        print("\n[XGBoost] Cargando pkl existente (reentrenar_xgboost=false)")

    gs     = joblib.load(ruta_gs)
    modelo = joblib.load(ruta_model)
    metricas = calcular_metricas("XGBoost (sin nucleo estructural)", modelo, X_te, y_te, ruta_reports)
    return modelo, gs, metricas


def train_lightgbm(datos_sup, config, ruta_models, ruta_reports, reentrenar):
    X_tr  = datos_sup['X_tr_sup']
    X_val = datos_sup['X_val_sup']
    X_te  = datos_sup['X_te_sup']
    y_tr  = datos_sup['y_tr']
    y_val = datos_sup['y_val']
    y_te  = datos_sup['y_te']

    ruta_gs    = f"{ruta_models}gs_LightGBM.pkl"
    ruta_model = f"{ruta_models}modelo_LightGBM.pkl"
    cfg = config.lightgbm

    if reentrenar or not (os.path.exists(ruta_gs) and os.path.exists(ruta_model)):
        print("\n[LightGBM] Iniciando GridSearchCV...")

        # Grilla base (los opcionales se agregan abajo si estan en el YAML)
        param_grid = {
            'n_estimators':      list(cfg.n_estimators_grid),
            'max_depth':         list(cfg.max_depth_grid),
            'learning_rate':     list(cfg.learning_rate_grid),
            'subsample':         list(cfg.subsample_grid),
            'colsample_bytree':  list(cfg.colsample_bytree_grid),
            'min_child_samples': list(cfg.min_child_samples_grid),
            'reg_alpha':         list(cfg.reg_alpha_grid),
        }
        # Parametros opcionales: solo entran si existen en el YAML
        if 'num_leaves_grid' in cfg:
            param_grid['num_leaves'] = list(cfg.num_leaves_grid)
        if 'reg_lambda_grid' in cfg:
            param_grid['reg_lambda'] = list(cfg.reg_lambda_grid)

        n_combinaciones = 1
        for v in param_grid.values():
            n_combinaciones *= len(v)
        print(f"  Combinaciones: {n_combinaciones} x {cfg.cv_folds} pliegues = {n_combinaciones * cfg.cv_folds} fits")

        cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
        gs = GridSearchCV(
            estimator=LGBMClassifier(random_state=cfg.random_state, n_jobs=-1, verbose=-1),
            param_grid=param_grid, scoring=cfg.scoring, cv=cv, n_jobs=-1, verbose=1
        )
        gs.fit(X_tr, y_tr)
        joblib.dump(gs, ruta_gs)

        best_params = gs.best_params_.copy()
        best_params.pop('n_estimators', None)
        print(f"\n[LightGBM] Mejores parametros: {gs.best_params_}")
        print(f"[LightGBM] Mejor F1 CV: {gs.best_score_:.4f}")
        print("[LightGBM] Entrenando modelo final con early stopping...")

        import lightgbm as lgb
        modelo = LGBMClassifier(
            **best_params,
            n_estimators=cfg.n_estimators_final,
            random_state=cfg.random_state,
            n_jobs=-1,
            verbose=-1
        )
        modelo.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=cfg.early_stopping_rounds, verbose=True)]
        )
        joblib.dump(modelo, ruta_model)
        print(f"[LightGBM] Modelos guardados en: {ruta_models}")
    else:
        print("\n[LightGBM] Cargando pkl existente (reentrenar_lightgbm=false)")

    gs     = joblib.load(ruta_gs)
    modelo = joblib.load(ruta_model)
    metricas = calcular_metricas("LightGBM (sin nucleo estructural)", modelo, X_te, y_te, ruta_reports)
    return modelo, gs, metricas


def run(datos_sup, config, reentrenar_xgb, reentrenar_lgbm):
    ruta_models  = config.paths.models
    ruta_reports = config.paths.reports

    modelo_xgb,  gs_xgb,  met_xgb  = train_xgboost(
        datos_sup, config, ruta_models, ruta_reports, reentrenar_xgb
    )
    modelo_lgbm, gs_lgbm, met_lgbm = train_lightgbm(
        datos_sup, config, ruta_models, ruta_reports, reentrenar_lgbm
    )

    comparativa = pd.DataFrame([met_xgb, met_lgbm])
    print(f"\n{'='*55}")
    print("TABLA COMPARATIVA — SUPERVISADOS")
    print(comparativa.to_string(index=False))
    comparativa.to_csv(f"{ruta_reports}metricas_supervisados.csv", index=False)

    return {
        'xgboost':  modelo_xgb,
        'lightgbm': modelo_lgbm,
        'gs_xgb':   gs_xgb,
        'gs_lgbm':  gs_lgbm,
        'metricas': comparativa,
    }
