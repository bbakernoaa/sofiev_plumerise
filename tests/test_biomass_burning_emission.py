import numpy as np
import xarray as xr
from sofiev_model.biomass_burning_emission import UFSCATChemFireGenerator

def test_instantiate_generator():
    """Tests if the UFSCATChemFireGenerator class can be instantiated."""
    try:
        UFSCATChemFireGenerator()
    except Exception as e:
        assert False, f"Failed to instantiate UFSCATChemFireGenerator: {e}"

def test_aggregate_raw_data_coarsen():
    """Tests the coarsen-based aggregation."""
    fire_sys = UFSCATChemFireGenerator()

    # Create a sample dataset with a size that is a multiple of the coarsen_factor
    coarsen_factor = 2
    lats = np.arange(0, 4)
    lons = np.arange(0, 4)

    # IGBP: Create a pattern where '3' is the mode
    igbp_data = np.array([
        [1, 3, 1, 1],
        [3, 3, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    ds_500m = xr.Dataset(
        {
            'FRP': (['lat', 'lon'], np.ones((4, 4))),
            'LAI': (['lat', 'lon'], np.full((4, 4), 0.5)),
            'IGBP': (['lat', 'lon'], igbp_data)
        },
        coords={'lat': lats, 'lon': lons}
    )

    # Aggregate the data
    ds_agg = fire_sys.aggregate_raw_data(ds_500m, coarsen_factor=coarsen_factor)

    # --- Verification ---
    # Check dimensions
    assert 'lat' in ds_agg.coords and 'lon' in ds_agg.coords
    assert len(ds_agg['lat']) == 2
    assert len(ds_agg['lon']) == 2

    # Check aggregated values
    # FRP (sum): Each 2x2 block has a sum of 4
    expected_frp = np.full((2, 2), 4.0)
    np.testing.assert_allclose(ds_agg['FRP'].values, expected_frp)

    # LAI (mean): The mean of 0.5 is 0.5
    expected_lai = np.full((2, 2), 0.5)
    np.testing.assert_allclose(ds_agg['LAI'].values, expected_lai)

    # IGBP (mode): The mode of the top-left 2x2 block is 3
    assert ds_agg['IGBP'].values[0, 0] == 3
