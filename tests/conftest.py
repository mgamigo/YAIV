from pathlib import Path
import warnings
import matplotlib
import matplotlib.pyplot as plt
import pytest
import numpy as np

# Use a headless backend for all plot tests
matplotlib.use("Agg")
# Make plots reproducible across machines
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "figure.figsize": (4, 3),
        "savefig.dpi": 100,
        "font.size": 10,
        "font.family": "DejaVu Sans",  # default available font
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.2,
    }
)


@pytest.fixture(scope="session")
def data_dir():
    """Root folder for external test data."""
    d = Path(__file__).parent / "data"
    if not d.exists():
        pytest.skip("tests/data folder not found", allow_module_level=True)
    return d


@pytest.fixture
def require():
    """Skip the current test if the given path does not exist."""

    def _require(path, reason=None):
        if not Path(path).exists():
            pytest.skip(reason or f"Missing test data: {path}")

    return _require


@pytest.fixture(autouse=True)
def _close_figs():
    """
    Auto-close figures between tests to avoid leakage
    """
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def quantity_allclose(actual, expected, **kwargs):
    expected = expected.to(actual.units)
    np.testing.assert_allclose(actual.magnitude, expected.magnitude, **kwargs)
    assert actual.units == expected.units

