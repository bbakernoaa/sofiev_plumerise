import pytest
import xarray as xr
import numpy as np
import pandas as pd
import dask.array as da
import xgboost as xgb
from unittest.mock import MagicMock
from sofiev_model.ufscat_fire_generator import FireEmissionGenerator

@pytest.fixture
def mock_fire_emission_generator(tmp_path):
    """Fixture to create a FireEmissionGenerator with a valid, mocked model."""
    # Create and save a minimal, valid XGBoost model
    model_path = tmp_path / "dummy_model.json"
    dummy_model = xgb.XGBRegressor(n_estimators=1, objective='reg:squarederror')
    # The model needs to be fit before it can be saved.
    dummy_model.fit(np.random.rand(2, 6), np.random.rand(2))
    dummy_model.save_model(model_path)

    # Create a dummy climatology NetCDF file
    climo_path = tmp_path / "dummy_climo.nc"
    climo_ds = xr.Dataset(
        {'emissions': (('month', 'lat', 'lon'), da.ones((12, 10, 10), chunks=(12, 5, 5)))},
        coords={'month': range(1, 13), 'lat': np.arange(10), 'lon': np.arange(10)}
    )
    climo_ds.to_netcdf(climo_path)

    # Initialize the generator, which will now load the valid model
    generator = FireEmissionGenerator(model_path=str(model_path), climo_path=str(climo_path))

    # Still mock the predict method to control the output for the test
    # It should return a numpy array, as the real predict method does.
    generator.model.predict = MagicMock(return_value=np.ones(100))
    return generator

def test_run_step_dask_awareness(mock_fire_emission_generator):
    """
    Test that run_step processes Dask-backed xarray objects without triggering
    computation and produces a Dask-backed output.
    """
    # 1. Create Dask-backed input DataArrays
    lats, lons = np.arange(10), np.arange(10)
    time = pd.to_datetime(['2023-07-01'])
    coords = {'lat': lats, 'lon': lons, 'time': time}

    # Use dask arrays for the data
    ufs_met = xr.Dataset({
        't2m': (('time', 'lat', 'lon'), da.full((1, 10, 10), 295.0, chunks=(1, 5, 5))),
        'rh2m': (('time', 'lat', 'lon'), da.full((1, 10, 10), 60.0, chunks=(1, 5, 5))),
        'u10': (('time', 'lat', 'lon'), da.full((1, 10, 10), 5.0, chunks=(1, 5, 5))),
        'v10': (('time', 'lat', 'lon'), da.full((1, 10, 10), 5.0, chunks=(1, 5, 5))),
        'precip': (('time', 'lat', 'lon'), da.zeros((1, 10, 10), chunks=(1, 5, 5))),
    }, coords=coords).squeeze()

    prev_states = xr.Dataset({
        'ffmc': (('lat', 'lon'), da.full((10, 10), 85.0, chunks=(5, 5))),
        'dmc': (('lat', 'lon'), da.full((10, 10), 15.0, chunks=(5, 5))),
        'dc': (('lat', 'lon'), da.full((10, 10), 200.0, chunks=(5, 5))),
    }, coords={'lat': lats, 'lon': lons})

    memory_grid = xr.DataArray(da.ones((10, 10), chunks=(5, 5)), coords=[lats, lons], dims=['lat', 'lon'])
    igbp_map = xr.DataArray(da.ones((10, 10), chunks=(5, 5)), coords=[lats, lons], dims=['lat', 'lon'])

    # 2. Execute the run_step
    emissions, new_states = mock_fire_emission_generator.run_step(ufs_met, prev_states, memory_grid, igbp_map)

    # 3. Assertions
    # Check that the output is an xarray DataArray with a Dask array
    assert isinstance(emissions, xr.DataArray)
    assert hasattr(emissions.data, 'dask')

    # Check that the new states are also Dask-backed
    assert isinstance(new_states, xr.Dataset)
    assert hasattr(new_states['dc'].data, 'dask')

    # Check the shape of the output
    assert emissions.shape == (10, 10)
    assert new_states['dc'].shape == (10, 10)

    # Check for history attribute
    assert 'history' in emissions.attrs

    # Check that computation has not been triggered
    assert emissions.chunks is not None

    # Trigger computation and check the result
    computed_emissions = emissions.compute()
    assert isinstance(computed_emissions.data, np.ndarray)
    assert computed_emissions.shape == (10, 10)
