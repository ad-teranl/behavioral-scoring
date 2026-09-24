"""
congelar_transformador.py
=========================
Se ejecuta UNA SOLA VEZ (y train_pipeline.py lo repite solo al reentrenar).

Guarda en models/artefactos_scoring.pkl los escaladores y codificadores
ajustados en el entrenamiento 2017-2019, la particion de cada operacion
y el cluster de riesgo de K-Means. Desde entonces, los cortes nuevos
solo se transforman: nunca se vuelve a ajustar nada.

Uso:  python congelar_transformador.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from hydra import utils
from omegaconf import DictConfig

from src import artefactos, preprocessing


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(config: DictConfig):
    root = utils.get_original_cwd() + "/"
    config.data.path = root + config.data.path
    config.paths.models = root + config.paths.models
    # Solo el CSV original: es el que definio el entrenamiento
    datasets = preprocessing.run_preprocessing(config)
    artefactos.guardar(artefactos.construir(datasets, config), config)


if __name__ == "__main__":
    main()
