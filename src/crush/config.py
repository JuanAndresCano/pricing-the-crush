"""Rutas y constantes globales del proyecto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # eventos ya armados (pre/post.parquet)
CONTROL_DIR = DATA_DIR / "control"  # cadenas del activo de control (SPY), compartidas entre eventos
PLACEBO_DIR = DATA_DIR / "placebo"  # ventanas placebo sin anuncio (experimento E4)
RAW_API_DIR = DATA_DIR / "raw_api"  # respuestas crudas de cada API, tal como llegaron
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"

# Convención de la teoría: T en años de tiempo de negociación (días hábiles / 252).
TRADING_DAYS_PER_YEAR = 252
