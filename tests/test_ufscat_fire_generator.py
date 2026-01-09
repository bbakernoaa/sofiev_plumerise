import os
import pytest
import numpy as np
import xarray as xr
import pandas as pd
import xgboost as xgb
from sofiev_model.ufscat_fire_generator import FireEmissionGenerator

# Helper function to create a dummy XGBoost model file
@pytest.fixture(scope="module")
def xgb_model_path(tmpdir_factory):
    """Creates a dummy XGBoost model file for testing."""
    model_path = str(tmpdir_factory.mktemp("data").join("test_model.json"))
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1)
    # Create dummy data to fit the model
    X = np.random.rand(10, 6)
    y = np.random.rand(10)
    model.fit(X, y)
    model.get_booster().save_model(model_path)
    return model_path

# Helper function to create a dummy climatology file
@pytest.fixture(scope="module")
def climo_path(tmpdir_factory):
    """Creates a dummy climatology NetCDF file."""
    climo_file = str(tmpdir_factory.mktemp("data").join("climo.nc"))
    ds = xr.Dataset(
        {'emissions': (('month', 'lat', 'lon'), np.ones((12, 180, 360)))},
        coords={
            'month': np.arange(1, 13),
            'lat': np.linspace(-89.5, 89.5, 180),
            'lon': np.linspace(-179.5, 179.5, 360)
        }
    )
    ds.to_netcdf(climo_file)
    return climo_file

def test_fire_emission_generator_init(xgb_model_path, climo_path):
    """Test the initialization of the FireEmissionGenerator."""
    generator = FireEmissionGenerator(model_path=xgb_model_path, climo_path=climo_path, target_res=1.0)
    assert isinstance(generator.model, xgb.XGBRegressor)
    assert isinstance(generator.climo, xr.Dataset)
    assert 'emissions' in generator.climo
    # Check if climatology is loaded lazily
    assert generator.climo.chunks is not None

def test_run_step(xgb_model_path, climo_path, tmpdir):
    """Test the core run_step method."""
    # 1. Setup
    # Create a local climo file with correct dimensions for this test
    local_climo_path = str(tmpdir.join("local_climo.nc"))
    ds = xr.Dataset(
        {'emissions': (('month', 'lat', 'lon'), np.ones((12, 2, 2)))},
        coords={
            'month': np.arange(1, 13),
            'lat': np.array([40.0, 41.0]),
            'lon': np.array([-100.0, -99.0])
        }
    )
    ds.to_netcdf(local_climo_path)

    generator = FireEmissionGenerator(model_path=xgb_model_path, climo_path=local_climo_path, target_res=1.0)

    # 2. Create mock inputs
    coords = {
        'time': [pd.Timestamp('2023-01-01')],
        'lat': np.array([40.0, 41.0]),
        'lon': np.array([-100.0, -99.0])
    }
    dims = ('time', 'lat', 'lon')

    ufs_met = xr.Dataset({
        't2m': (dims, np.full((1, 2, 2), 293.15)), # 20C
        'rh2m': (dims, np.full((1, 2, 2), 50.0)),
        'u10': (dims, np.full((1, 2, 2), 5.0)),
        'v10': (dims, np.full((1, 2, 2), 5.0)),
        'precip': (dims, np.full((1, 2, 2), 0.1)),
    }, coords=coords)

    prev_states = xr.Dataset({
        'ffmc': (('lat', 'lon'), np.full((2, 2), 85.0)),
        'dmc': (('lat', 'lon'), np.full((2, 2), 50.0)),
        'dc': (('lat', 'lon'), np.full((2, 2), 300.0)),
    }, coords={'lat': coords['lat'], 'lon': coords['lon']})

    memory_grid = xr.DataArray(np.zeros((2, 2)), coords={'lat': coords['lat'], 'lon': coords['lon']}, name='frp_memory')
    igbp_map = xr.DataArray(np.full((2, 2), 4), coords={'lat': coords['lat'], 'lon': coords['lon']}, name='igbp_class')

    # 3. Execute
    emissions, new_states = generator.run_step(ufs_met.isel(time=0), prev_states, memory_grid, igbp_map)

    # 4. Verify outputs
    assert isinstance(emissions, xr.DataArray)
    assert isinstance(new_states, xr.Dataset)

    # Check coordinates and dimensions
    assert 'lat' in emissions.coords and 'lon' in emissions.coords
    assert emissions.shape == (2, 2)
    assert set(new_states.data_vars) == {'ffmc', 'dmc', 'dc'}
    assert new_states['ffmc'].shape == (2, 2)

    # Check for provenance attribute
    assert 'history' in emissions.attrs
    assert "UFSCATChemFireGenerator" in emissions.attrs['history']

def test_save_load_state(tmpdir, xgb_model_path, climo_path):
    """Test saving and loading the FWI state."""
    generator = FireEmissionGenerator(model_path=xgb_model_path, climo_path=climo_path)
    state_file = str(tmpdir.join("fwi_state.nc"))

    original_state = xr.Dataset({
        'ffmc': (('lat', 'lon'), np.random.rand(10, 10)),
        'dmc': (('lat', 'lon'), np.random.rand(10, 10)),
        'dc': (('lat', 'lon'), np.random.rand(10, 10)),
    }, coords={'lat': np.arange(10), 'lon': np.arange(10)})

    generator.save_state(original_state, state_file)
    assert os.path.exists(state_file)

    loaded_state = generator.load_state(state_file)
    xr.testing.assert_allclose(original_state, loaded_state)
