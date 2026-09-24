"""
actualizar_corte.py
===================
Incorpora un corte mensual nuevo desde la consola (alternativa al boton del tablero).

Uso:  python actualizar_corte.py ruta\\al\\corte_enero_2020.csv

1. Valida columnas, formato de fecha y que el corte sea posterior al historico
2. Lo guarda en data/raw/cortes_nuevos/ (el CSV original nunca se modifica)
3. Regenera reports/scoring/ con los modelos y el transformador congelados
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src import scoring_cartera
from src.config_local import cargar_config


def main():
    if len(sys.argv) != 2:
        print("Uso: python actualizar_corte.py ruta_del_corte.csv")
        sys.exit(1)
    config = cargar_config()
    nuevo = pd.read_csv(sys.argv[1], sep=config.data.sep)
    errores, avisos = scoring_cartera.validar_corte(nuevo, config)
    for a in avisos:
        print("AVISO:", a)
    if errores:
        for e in errores:
            print("ERROR:", e)
        sys.exit(1)
    ruta = scoring_cartera.incorporar_corte(nuevo, config)
    print(f"Corte guardado en {ruta}")
    scoring_cartera.run(config)


if __name__ == "__main__":
    main()
