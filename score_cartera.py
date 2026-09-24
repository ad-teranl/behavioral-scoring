"""
score_cartera.py
================
Punto de entrada del scoring de cartera (paso previo al dashboard).

Uso (siempre desde la raiz del proyecto, en consola):
  python score_cartera.py

Requiere que existan los pkl en models/ (generados por train_pipeline.py).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra
from hydra import utils
from omegaconf import DictConfig

import src.scoring_cartera as scoring_cartera


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(config: DictConfig):
    root = utils.get_original_cwd() + "/"
    config.data.path = root + config.data.path
    config.paths.models = root + config.paths.models
    config.paths.reports = root + config.paths.reports
    scoring_cartera.run(config)


if __name__ == "__main__":
    main()
