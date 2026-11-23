import crush
from crush import config


def test_package_imports():
    assert crush.__version__


def test_trading_days_convention():
    assert config.TRADING_DAYS_PER_YEAR == 252


def test_project_paths_point_inside_repo():
    assert config.DATA_DIR.parent == config.ROOT
    assert config.RESULTS_DIR.parent == config.ROOT
