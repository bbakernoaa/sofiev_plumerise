# Tests for the FireEmissionGenerator class
import os
import pytest
import numpy as np
import xarray as xr
import pandas as pd
from sofiev_model.ufscat_fire_generator import FireEmissionGenerator

@pytest.fixture(scope="module")
def dummy_model_path(tmpdir_factory):
    """Creates a dummy XGBoost model file."""
    fn = tmpdir_factory.mktemp("data").join("dummy_model.json")
    with open(fn, "w") as f:
        f.write("{}")  # Minimal valid JSON
    return str(fn)

@pytest.fixture(scope="module")
def dummy_climo_path(tmpdir_factory):
    """Creates a dummy climatology NetCDF file with a small grid."""
    fn = tmpdir_factory.mktemp("data").join("dummy_climo.nc")
    ds = xr.Dataset(
        {"emissions": (("month", "lat", "lon"), np.ones((12, 45, 90)))},
        coords={
            "month": np.arange(1, 13),
            "lat": np.linspace(-90, 90, 45),
            "lon": np.linspace(-180, 180, 90),
        },
    )
    ds.to_netcdf(fn, engine="netcdf4")
    return str(fn)

@pytest.fixture
def fire_generator(dummy_model_path, dummy_climo_path, monkeypatch):
    """Initializes the FireEmissionGenerator with a mocked XGBoost model."""
    class MockXGB:
        def load_model(self, *args, **kwargs):
            """Mock load_model to do nothing."""
            pass
        def predict(self, data):
            """Mock predict to return a neutral scaling factor."""
            return np.ones(data.shape[0])

    # Use monkeypatch to replace the real XGBRegressor with our mock
    # This ensures that the real `load_model` is never called on our dummy file.
    monkeypatch.setattr("xgboost.XGBRegressor", MockXGB)

    # Initialize the generator with a resolution that matches the test data (4x4 degrees)
    generator = FireEmissionGenerator(
        model_path=dummy_model_path,
        climo_path=dummy_climo_path,
        target_res=4.0
    )
    return generator

def test_init_lazy_loading(fire_generator):
    """
    Tests if the climatology dataset is loaded lazily with dask.
    """
    assert hasattr(fire_generator.climo['emissions'].data, 'dask'), "Emissions data array should be Dask-backed."

def test_save_and_load_state_provenance(fire_generator, tmpdir):
    """
    Tests that save_state adds a history attribute and load_state reads it.
    """
    state_file = os.path.join(str(tmpdir), "test_state.nc")
    dummy_states = {
        'ffmc': np.full((45, 90), 85.0),
        'dmc': np.full((45, 90), 6.0),
        'dc': np.full((45, 90), 15.0),
    }

    # 1. Save state and check for history attribute
    fire_generator.save_state(dummy_states, state_file)
    with xr.open_dataset(state_file) as ds:
        assert "history" in ds.attrs
        assert "Created by Aero" in ds.attrs["history"]

    # 2. Load state and confirm data integrity
    loaded_states = fire_generator.load_state(state_file)
    for key in dummy_states:
        np.testing.assert_array_equal(dummy_states[key], loaded_states[key])

def test_run_step_logic(fire_generator):
    """
    Tests the basic execution flow of a single timestep.
    """
    # Create dummy inputs
    ufs_met = xr.Dataset(
        {
            "t2m": (("time", "lat", "lon"), np.full((1, 45, 90), 293.15)), # 20C
            "rh2m": (("time", "lat", "lon"), np.full((1, 45, 90), 50.0)),
            "u10": (("time", "lat", "lon"), np.full((1, 45, 90), 5.0)),
            "v10": (("time", "lat", "lon"), np.full((1, 45, 90), 5.0)),
            "precip": (("time", "lat", "lon"), np.zeros((1, 45, 90))),
        },
        coords={
            "time": [pd.to_datetime("2023-01-15")],
            "lat": np.linspace(-90, 90, 45),
            "lon": np.linspace(-180, 180, 90),
        }
    )
    prev_states = {
        'ffmc': np.full((45, 90), 85.0),
        'dmc': np.full((45, 90), 6.0),
        'dc': np.full((45, 90), 15.0),
    }
    memory_grid = np.zeros((45, 90))
    igbp_map = np.ones((45, 90))

    # Execute the step
    final_emissions, new_states = fire_generator.run_step(ufs_met, prev_states, memory_grid, igbp_map)

    # Validate outputs
    assert final_emissions.shape == (45, 90)
    # Since the mock model predicts 1.0, and smoothing/clipping occurs,
    # the output should be the climatology * a smoothed factor near 1.
    expected_emissions = fire_generator.climo['emissions'].sel(month=1).values
    # Gaussian filter will smooth the edges, so we can't do an exact match.
    # Check that the mean is close to the expected value.
    assert np.allclose(np.mean(final_emissions), np.mean(expected_emissions), rtol=0.1)

    assert 'ffmc' in new_states and 'dmc' in new_states and 'dc' in new_states
    assert new_states['ffmc'].shape == (45, 90)

