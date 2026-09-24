"""
config_local.py
Carga config/main.yaml con rutas absolutas SIN Hydra.
Lo usan el dashboard y actualizar_corte.py (no necesitan la magia de Hydra).
"""
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def cargar_config():
    cfg = OmegaConf.load(ROOT / "config" / "main.yaml")
    cfg.data.path = str(ROOT / cfg.data.path)
    cfg.paths.models = str(ROOT / cfg.paths.models) + "/"
    cfg.paths.reports = str(ROOT / cfg.paths.reports) + "/"
    return cfg
