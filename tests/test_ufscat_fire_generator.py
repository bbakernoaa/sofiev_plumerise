import pytest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import MagicMock
from sofiev_model.ufscat_fire_generator import UFSCATChemFireGenerator


@pytest.fixture
def fire_generator_instance(monkeypatch):
    """
    Provides a UFSCATChemFireGenerator instance for testing, with a mocked
    __init__ to avoid file I/O.
    """
    # Mock the __init__ method to bypass loading of XGBoost model and climatology
    def mock_init(self, model_path, climo_path, target_res=1.0):
        self.res = target_res
        self.target_lats = np.arange(-90 + self.res / 2, 90, self.res)
        self.target_lons = np.arange(-180 + self.res / 2, 180, self.res)
        self.fwi_engine = MagicMock()
        self.model = MagicMock()
        self.climo = xr.Dataset(
            {
                "emissions": (
                    ("month", "lat", "lon"),
                    np.ones((12, len(self.target_lats), len(self.target_lons))),
                )
            },
            coords={
                "month": np.arange(1, 13),
                "lat": self.target_lats,
                "lon": self.target_lons,
            },
        )

    monkeypatch.setattr(UFSCATChemFireGenerator, "__init__", mock_init)
    return UFSCATChemFireGenerator("dummy_model.json", "dummy_climo.nc")


def test_calculate_vpd(fire_generator_instance):
    """
    Tests the `calculate_vpd` method with known values to ensure correctness.
    """
    # Test cases: [T_kelvin, RH_percent]
    test_cases = np.array([
        [273.15, 50.0],  # 0°C
        [283.15, 60.0],  # 10°C
        [293.15, 70.0],  # 20°C
    ])
    t2m = test_cases[:, 0]
    rh2m = test_cases[:, 1]

    # Expected VPD in hPa, calculated independently
    expected_vpd = np.array([3.056, 4.908, 7.009])

    # Calculate VPD using the method
    calculated_vpd = fire_generator_instance.calculate_vpd(t2m, rh2m)

    # Assert that the calculated values are close to the expected values
    np.testing.assert_allclose(calculated_vpd, expected_vpd, rtol=1e-3)


def test_run_step(fire_generator_instance):
    """
    Tests the `run_step` method to ensure it correctly processes inputs
    and produces valid outputs.
    """
    # 1. Setup Mock Inputs
    n_lat = len(fire_generator_instance.target_lats)
    n_lon = len(fire_generator_instance.target_lons)
    shape = (n_lat, n_lon)

    ufs_met = xr.Dataset(
        {
            "t2m": (("lat", "lon"), np.full(shape, 293.15)),
            "rh2m": (("lat", "lon"), np.full(shape, 70.0)),
            "u10": (("lat", "lon"), np.full(shape, 5.0)),
            "v10": (("lat", "lon"), np.full(shape, 5.0)),
            "precip": (("lat", "lon"), np.zeros(shape)),
        },
        coords={
            "lat": fire_generator_instance.target_lats,
            "lon": fire_generator_instance.target_lons,
            "time": [pd.to_datetime("2023-01-15")],
        },
    )
    prev_states = {
        "ffmc": np.full(shape, 85.0),
        "dmc": np.full(shape, 60.0),
        "dc": np.full(shape, 400.0),
    }
    memory_grid = np.zeros(shape)
    igbp_map = np.full(shape, 10)

    # 2. Configure Mock Return Values
    fire_generator_instance.fwi_engine.calculate_ffmc.return_value = np.full(shape, 87.0)
    fire_generator_instance.fwi_engine.calculate_dmc.return_value = np.full(shape, 62.0)
    fire_generator_instance.fwi_engine.calculate_dc.return_value = np.full(shape, 402.0)
    fire_generator_instance.fwi_engine.calculate_bui.return_value = np.full(shape, 65.0)
    fire_generator_instance.model.predict.return_value = np.full(shape[0] * shape[1], 1.5)

    # 3. Execute the Method
    emissions, new_states = fire_generator_instance.run_step(
        ufs_met, prev_states, memory_grid, igbp_map
    )

    # 4. Assertions
    assert emissions.shape == shape
    assert np.all(emissions >= 0)
    assert "ffmc" in new_states
    assert "dmc" in new_states
    assert "dc" in new_states
    assert new_states["ffmc"].shape == shape
    fire_generator_instance.model.predict.assert_called_once()
