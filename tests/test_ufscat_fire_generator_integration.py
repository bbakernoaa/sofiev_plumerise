import os
import pytest
import xarray as xr
import numpy as np
import pandas as pd
import xgboost as xgb
from sofiev_model.ufscat_fire_generator import FireEmissionGenerator


@pytest.fixture(scope="module")
def project_root():
    """Fixture to provide the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def setup_test_data(tmp_path):
    """Prepares all necessary dummy data files for an integration test run."""
    # Define grid dimensions
    lats = np.arange(40, 40.2, 0.04, dtype=np.float32)
    lons = np.arange(-105.2, -105, 0.04, dtype=np.float32)
    time = pd.to_datetime(["2023-07-01T12:00:00"])
    coords_2d = {"lat": lats, "lon": lons}
    coords_3d = {"time": time, "lat": lats, "lon": lons}

    # 1. Dummy XGBoost Model
    model = xgb.XGBRegressor(objective="reg:squarederror")
    dummy_X = np.random.rand(1, 6)
    dummy_y = np.random.rand(1)
    model.fit(dummy_X, dummy_y)
    model_path = tmp_path / "dummy_model.json"
    model.save_model(model_path)

    # 2. Dummy GBBEPx Climatology
    climo_path = tmp_path / "dummy_climo.nc"
    climo_ds = xr.Dataset(
        {"emissions": (("month", "lat", "lon"), np.ones((12, len(lats), len(lons))))},
        coords={"month": range(1, 13), "lat": lats, "lon": lons},
    )
    climo_ds.to_netcdf(climo_path)

    # 3. Dummy IGBP Map
    igbp_path = tmp_path / "dummy_igbp.nc"
    igbp_ds = xr.Dataset(
        {"band_1": (("lat", "lon"), np.full((len(lats), len(lons)), 4, dtype=np.int8))},
        coords=coords_2d,
    )
    igbp_ds.to_netcdf(igbp_path)

    # 4. Dummy UFS Meteorology
    met_path = tmp_path / "dummy_met.nc"
    met_ds = xr.Dataset(
        {
            "t2m": (
                ("time", "lat", "lon"),
                np.full((1, len(lats), len(lons)), 298.0, dtype=np.float32),
            ),
            "rh2m": (
                ("time", "lat", "lon"),
                np.full((1, len(lats), len(lons)), 45.0, dtype=np.float32),
            ),
            "u10": (
                ("time", "lat", "lon"),
                np.full((1, len(lats), len(lons)), 3.0, dtype=np.float32),
            ),
            "v10": (
                ("time", "lat", "lon"),
                np.full((1, len(lats), len(lons)), 3.0, dtype=np.float32),
            ),
            "precip": (
                ("time", "lat", "lon"),
                np.zeros((1, len(lats), len(lons)), dtype=np.float32),
            ),
        },
        coords=coords_3d,
    )
    met_ds.to_netcdf(met_path)

    # 5. Dummy Previous FWI States
    states_path = tmp_path / "dummy_prev_states.nc"
    states_ds = xr.Dataset(
        {
            "ffmc": (
                ("lat", "lon"),
                np.full((len(lats), len(lons)), 88.0, dtype=np.float32),
            ),
            "dmc": (
                ("lat", "lon"),
                np.full((len(lats), len(lons)), 90.0, dtype=np.float32),
            ),
            "dc": (
                ("lat", "lon"),
                np.full((len(lats), len(lons)), 350.0, dtype=np.float32),
            ),
        },
        coords=coords_2d,
    )
    states_ds.to_netcdf(states_path)

    # 6. Dummy Fire Memory Grid
    memory_path = tmp_path / "dummy_memory.nc"
    memory_ds = xr.Dataset(
        {"FRP": (("lat", "lon"), np.ones((len(lats), len(lons)), dtype=np.float32))},
        coords=coords_2d,
    )
    memory_ds.to_netcdf(memory_path)

    return {
        "model_path": str(model_path),
        "climo_path": str(climo_path),
        "igbp_path": str(igbp_path),
        "met_path": str(met_path),
        "states_path": str(states_path),
        "memory_path": str(memory_path),
        "lats": lats,
        "lons": lons,
    }


def test_fire_emission_generator_integration(setup_test_data):
    """
    Tests the end-to-end workflow of the FireEmissionGenerator class.
    This test creates a complete set of realistic, albeit dummy, input
    data files in a temporary directory. It then initializes the
    generator, runs a single time step, and validates the outputs and
    the state-saving/loading mechanism.
    """
    paths = setup_test_data

    # 1. Initialize the generator
    generator = FireEmissionGenerator(
        model_path=paths["model_path"], climo_path=paths["climo_path"]
    )

    # 2. Load the prepared data
    ufs_met = xr.open_dataset(paths["met_path"]).squeeze()
    prev_states = xr.open_dataset(paths["states_path"])
    memory_grid = xr.open_dataset(paths["memory_path"])["FRP"]
    igbp_map = xr.open_dataset(paths["igbp_path"])["band_1"]

    # 3. Run the core method
    final_emissions, new_states = generator.run_step(
        ufs_met=ufs_met,
        prev_states=prev_states,
        memory_grid=memory_grid,
        igbp_map=igbp_map,
    )

    # 4. Assertions on the output
    expected_shape = (len(paths["lats"]), len(paths["lons"]))
    assert final_emissions.shape == expected_shape, "Emission grid has incorrect shape"
    assert new_states["dc"].shape == expected_shape, "New DC state has incorrect shape"
    assert "history" in final_emissions.attrs, (
        "History attribute missing from emissions"
    )
    assert not np.isnan(final_emissions.values).any(), "NaN values found in emissions"
    # Verify that the output coordinates are preserved from the input meteorological data,
    # which is the source of the grid information for the calculations.
    xr.testing.assert_allclose(final_emissions.coords["lat"], ufs_met.coords["lat"])
    xr.testing.assert_allclose(final_emissions.coords["lon"], ufs_met.coords["lon"])

    # 5. Test state saving and loading
    output_states_path = os.path.join(
        os.path.dirname(paths["model_path"]), "new_fwi_states.nc"
    )
    generator.save_state(new_states, output_states_path)
    assert os.path.exists(output_states_path), "State file was not saved"

    loaded_states = generator.load_state(output_states_path)
    xr.testing.assert_allclose(new_states, loaded_states)

    # Test loading a non-existent file
    with pytest.raises(FileNotFoundError):
        generator.load_state("non_existent_file.nc")
