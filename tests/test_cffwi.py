import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

# Import the class to be tested
from sofiev_model.cffwi import FWI_Engine_Vectorized

# Define a common coordinate system for the tests
TEST_COORDS = {"lat": np.arange(4), "lon": np.arange(5)}
TEST_DIMS = ("lat", "lon")


@pytest.fixture
def sample_weather_data() -> xr.Dataset:
    """Provides a sample xr.Dataset of weather data for testing."""
    return xr.Dataset(
        {
            "temp": (TEST_DIMS, np.full((4, 5), 25.0)),
            "rh": (TEST_DIMS, np.full((4, 5), 60.0)),
            "wind": (TEST_DIMS, np.full((4, 5), 15.0)),
            "precip": (TEST_DIMS, np.zeros((4, 5))),
        },
        coords=TEST_COORDS,
    )


@pytest.fixture
def sample_fwi_states() -> xr.Dataset:
    """Provides a sample xr.Dataset of initial FWI states."""
    return xr.Dataset(
        {
            "ffmc_prev": (TEST_DIMS, np.full((4, 5), 85.0)),
            "dmc_prev": (TEST_DIMS, np.full((4, 5), 15.0)),
            "dc_prev": (TEST_DIMS, np.full((4, 5), 200.0)),
        },
        coords=TEST_COORDS,
    )


def test_calculate_ffmc_returns_dataarray(sample_weather_data, sample_fwi_states):
    """Test that calculate_ffmc returns an xr.DataArray with coords preserved."""
    result = FWI_Engine_Vectorized.calculate_ffmc(
        sample_weather_data["temp"],
        sample_weather_data["rh"],
        sample_weather_data["wind"],
        sample_weather_data["precip"],
        sample_fwi_states["ffmc_prev"],
    )
    assert isinstance(result, xr.DataArray)
    assert result.coords.keys() == sample_weather_data.coords.keys()
    assert result.shape == (4, 5)


def test_calculate_dmc_returns_dataarray(sample_weather_data, sample_fwi_states):
    """Test that calculate_dmc returns an xr.DataArray with coords preserved."""
    result = FWI_Engine_Vectorized.calculate_dmc(
        sample_weather_data["temp"],
        sample_weather_data["rh"],
        sample_weather_data["precip"],
        sample_fwi_states["dmc_prev"],
        month=7,
    )
    assert isinstance(result, xr.DataArray)
    assert result.coords.keys() == sample_weather_data.coords.keys()
    assert result.shape == (4, 5)


def test_calculate_dc_returns_dataarray(sample_weather_data, sample_fwi_states):
    """Test that calculate_dc returns an xr.DataArray with coords preserved."""
    result = FWI_Engine_Vectorized.calculate_dc(
        sample_weather_data["temp"],
        sample_weather_data["precip"],
        sample_fwi_states["dc_prev"],
        month=7,
    )
    assert isinstance(result, xr.DataArray)
    assert result.coords.keys() == sample_weather_data.coords.keys()
    assert result.shape == (4, 5)


def test_fwi_components_with_dask(sample_weather_data, sample_fwi_states):
    """Verify that FWI calculations remain lazy when using Dask chunks."""
    # Chunk the input data
    dask_weather = sample_weather_data.chunk({"lat": 2, "lon": 2})
    dask_states = sample_fwi_states.chunk({"lat": 2, "lon": 2})

    # Run calculations
    new_ffmc = FWI_Engine_Vectorized.calculate_ffmc(
        dask_weather["temp"],
        dask_weather["rh"],
        dask_weather["wind"],
        dask_weather["precip"],
        dask_states["ffmc_prev"],
    )
    new_dmc = FWI_Engine_Vectorized.calculate_dmc(
        dask_weather["temp"],
        dask_weather["rh"],
        dask_weather["precip"],
        dask_states["dmc_prev"],
        month=7,
    )
    new_dc = FWI_Engine_Vectorized.calculate_dc(
        dask_weather["temp"], dask_weather["precip"], dask_states["dc_prev"], month=7
    )

    # Check that the results are Dask-backed (lazy)
    assert hasattr(new_ffmc.data, "dask")
    assert hasattr(new_dmc.data, "dask")
    assert hasattr(new_dc.data, "dask")

    # Compute the results and check shapes
    computed_ffmc = new_ffmc.compute()
    computed_dmc = new_dmc.compute()
    computed_dc = new_dc.compute()

    assert computed_ffmc.shape == (4, 5)
    assert computed_dmc.shape == (4, 5)
    assert computed_dc.shape == (4, 5)


def test_fwi_full_calculation_consistency(sample_weather_data, sample_fwi_states):
    """Compare eager vs. lazy FWI results to ensure consistency."""
    # Eager computation
    eager_ffmc = FWI_Engine_Vectorized.calculate_ffmc(
        sample_weather_data["temp"],
        sample_weather_data["rh"],
        sample_weather_data["wind"],
        sample_weather_data["precip"],
        sample_fwi_states["ffmc_prev"],
    )
    eager_dmc = FWI_Engine_Vectorized.calculate_dmc(
        sample_weather_data["temp"],
        sample_weather_data["rh"],
        sample_weather_data["precip"],
        sample_fwi_states["dmc_prev"],
        month=7,
    )
    eager_dc = FWI_Engine_Vectorized.calculate_dc(
        sample_weather_data["temp"],
        sample_weather_data["precip"],
        sample_fwi_states["dc_prev"],
        month=7,
    )

    # Lazy (Dask) computation
    dask_weather = sample_weather_data.chunk({"lat": 2})
    dask_states = sample_fwi_states.chunk({"lat": 2})

    lazy_ffmc = FWI_Engine_Vectorized.calculate_ffmc(
        dask_weather["temp"],
        dask_weather["rh"],
        dask_weather["wind"],
        dask_weather["precip"],
        dask_states["ffmc_prev"],
    ).compute()
    lazy_dmc = FWI_Engine_Vectorized.calculate_dmc(
        dask_weather["temp"],
        dask_weather["rh"],
        dask_weather["precip"],
        dask_states["dmc_prev"],
        month=7,
    ).compute()
    lazy_dc = FWI_Engine_Vectorized.calculate_dc(
        dask_weather["temp"], dask_weather["precip"], dask_states["dc_prev"], month=7
    ).compute()

    # Use xarray's testing utility for robust comparison
    assert_allclose(eager_ffmc, lazy_ffmc)
    assert_allclose(eager_dmc, lazy_dmc)
    assert_allclose(eager_dc, lazy_dc)
