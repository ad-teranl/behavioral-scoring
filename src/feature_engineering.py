"""
feature_engineering.py
======================
Seleccion de variables para los modelos supervisados y no supervisados.

SUPERVISADOS:
  Las 11 variables del Escenario 3 ya estan fijadas en el YAML
  (criterio de deslinde contable — excluye nucleo circular de mora).
  Se toman directamente del config sin recalcular IV/WoE.

NO SUPERVISADOS:
  Las 18 variables estan fijadas en el YAML como interseccion
  MIUFS ∩ Autoencoder (fijadas para reproducibilidad, ver nota en QMD).
  Opcionalmente se recalculan MIUFS y autoencoder para verificacion.

Salida:
  Subsets seleccionados listos para entrar a los modelos.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew
from sklearn.preprocessing import StandardScaler


def select_features_supervisado(datasets, config):
    """
    Selecciona las 11 variables exogenas del YAML para los modelos supervisados.
    Opera sobre X_train_sup_bal (SMOTE), X_val_sup, X_test_sup.
    """
    features = list(config.features_supervisado)

    X_tr  = datasets['X_train_sup_bal'][features].copy()
    X_val = datasets['X_val_sup'][features].copy()
    X_te  = datasets['X_test_sup'][features].copy()
    y_tr  = datasets['y_train_bal']
    y_val = datasets['y_val']
    y_te  = datasets['y_test']

    print(f"\n[FE-Supervisado] {len(features)} variables seleccionadas del YAML")
    print(f"   Train (bal): {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")

    return {
        'X_tr_sup':  X_tr,
        'X_val_sup': X_val,
        'X_te_sup':  X_te,
        'y_tr':      y_tr,
        'y_val':     y_val,
        'y_te':      y_te,
    }


def select_features_nosupervisado(datasets, config):
    """
    Selecciona las 18 variables del YAML para los modelos no supervisados.
    Opera sobre X_train_nosup, X_val_nosup, X_test_nosup (sin SMOTE).
    """
    features = list(config.features_nosupervisado)

    X_tr  = datasets['X_train_nosup'][features].copy()
    X_val = datasets['X_val_nosup'][features].copy()
    X_te  = datasets['X_test_nosup'][features].copy()

    print(f"\n[FE-NoSupervisado] {len(features)} variables seleccionadas del YAML")
    print(f"   Train: {X_tr.shape} | Val: {X_val.shape} | Test: {X_te.shape}")

    return {
        'X_tr_nosup':  X_tr,
        'X_val_nosup': X_val,
        'X_te_nosup':  X_te,
        'y_tr':        datasets['y_train'],
        'y_val':       datasets['y_val'],
        'y_te':        datasets['y_test'],
    }


def run_miufs(X_train_nosup, tra_esc_nosup, config):
    """
    MIUFS: score compuesto (kurtosis, asimetria, entropia, varianza).
    Retorna lista ordenada de las top_n variables.
    """
    from scipy.stats import entropy as sp_entropy

    top_n = config.miufs.top_n
    pesos = np.array([
        config.miufs.peso_kurtosis,
        config.miufs.peso_asimetria,
        config.miufs.peso_entropia,
        config.miufs.peso_varianza,
    ])

    X = X_train_nosup[tra_esc_nosup].copy()
    metrics = pd.DataFrame(index=tra_esc_nosup,
                           columns=['Kurtosis', 'Asimetria', 'Entropia', 'Varianza'])

    for col in tra_esc_nosup:
        hist, _ = np.histogram(X[col], bins=10)
        prob = hist / hist.sum()
        metrics.loc[col, 'Kurtosis']  = kurtosis(X[col])
        metrics.loc[col, 'Asimetria'] = skew(X[col])
        metrics.loc[col, 'Entropia']  = sp_entropy(prob)
        metrics.loc[col, 'Varianza']  = np.var(X[col])

    metrics = metrics.astype(float)
    scaler  = StandardScaler()
    metrics_norm = pd.DataFrame(
        scaler.fit_transform(metrics),
        columns=metrics.columns,
        index=metrics.index
    )
    metrics_norm['Score'] = np.dot(metrics_norm.values, pesos)
    metrics_sorted = metrics_norm.sort_values('Score', ascending=False)

    return metrics_sorted.head(top_n).index.tolist()


def run_autoencoder(X_train_nosup, tra_esc_nosup, config):
    """
    CLFS: autoencoder para seleccion de variables no supervisadas.
    Retorna lista de las top_n variables por importancia en la capa oculta.
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam

    tf.keras.utils.set_random_seed(config.autoencoder.random_seed)

    X = X_train_nosup[tra_esc_nosup].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim   = X_scaled.shape[1]
    hidden_dim  = config.autoencoder.hidden_dim
    input_layer = Input(shape=(input_dim,))
    encoded     = Dense(hidden_dim, activation='relu')(input_layer)
    decoded     = Dense(input_dim, activation='linear')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(
        optimizer=Adam(learning_rate=config.autoencoder.learning_rate),
        loss='mse'
    )
    autoencoder.fit(
        X_scaled, X_scaled,
        epochs=config.autoencoder.epochs,
        batch_size=config.autoencoder.batch_size,
        verbose=0
    )

    encoder   = Model(input_layer, encoded)
    weights   = encoder.layers[1].get_weights()[0]
    importancia = np.mean(np.abs(weights), axis=1)

    metrics_clfs = pd.DataFrame({
        'Variable':   tra_esc_nosup,
        'Importancia': importancia
    }).sort_values('Importancia', ascending=False)

    top_n = config.miufs.top_n
    return metrics_clfs.head(top_n)['Variable'].tolist()
