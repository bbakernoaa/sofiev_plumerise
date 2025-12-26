import pytest
import numpy as np
import xarray as xr
import pandas as pd
from sofiev_model.ufscat_fire_generator import UFSCATChemFireGenerator

@pytest.fixture
def fire_generator_instance():
    """Provides a UFSCATChemFireGenerator instance for testing."""
    return UFSCATChemFireGenerator(target_res=1.0) # Coarse resolution for speed

@pytest.fixture
def mock_satellite_data():
    """Creates mock satellite data for testing."""
    lats = np.arange(40, 42, 0.5)
    lons = np.arange(-106, -104, 0.5)
    time = pd.to_datetime(['2023-01-01', '2023-02-01'])
    return xr.Dataset(
        {
            'FRP': (('lat', 'lon', 'time'), np.random.rand(len(lats), len(lons), len(time)) * 100),
            'LAI': (('lat', 'lon', 'time'), np.random.rand(len(lats), len(lons), len(time)) * 5),
            'IGBP': (('lat', 'lon'), np.random.randint(1, 17, size=(len(lats), len(lons)))),
        },
        coords={'lat': lats, 'lon': lons, 'time': time}
    )

@pytest.fixture
def mock_met_data(fire_generator_instance):
    """Creates mock meteorology data for testing."""
    lats = fire_generator_instance.target_lats
    lons = fire_generator_instance.target_lons
    time = pd.to_datetime(['2023-01-01', '2023-02-01'])
    return xr.Dataset(
        {
            'vpd': (('lat', 'lon', 'time'), np.random.rand(len(lats), len(lons), len(time)) * 30),
            'soil_m': (('lat', 'lon', 'time'), np.random.rand(len(lats), len(lons), len(time)) * 0.5),
        },
        coords={'lat': lats, 'lon': lons, 'time': time}
    )

def test_initialization(fire_generator_instance):
    """Tests if the class initializes correctly."""
    assert fire_generator_instance.res == 1.0
    assert len(fire_generator_instance.target_lats) > 0
    assert len(fire_generator_instance.target_lons) > 0

def test_aggregate_raw_data(fire_generator_instance, mock_satellite_data):
    """Tests the aggregation functionality."""
    ds_4km = fire_generator_instance.aggregate_raw_data(mock_satellite_data.isel(time=0))
    assert 'FRP' in ds_4km
    assert 'LAI' in ds_4km
    assert 'IGBP' in ds_4km
    assert ds_4km.FRP.shape == (len(fire_generator_instance.target_lats), len(fire_generator_instance.target_lons))

def test_generate_features(fire_generator_instance, mock_satellite_data, mock_met_data):
    """
    Tests the feature generation functionality, ensuring it uses the output
    from the aggregation step.
    """
    # 1. Run aggregation for each time slice
    ds_4km_slices = [fire_generator_instance.aggregate_raw_data(mock_satellite_data.isel(time=t))
                     for t in range(len(mock_satellite_data.time))]
    ds_4km_time = xr.concat(ds_4km_slices, dim=mock_satellite_data.time)

    # 2. Generate features using the aggregated data and mock meteorology
    df = fire_generator_instance.generate_features(ds_4km_time, mock_met_data)

    # 3. Validate the output
    assert not df.empty
    expected_cols = ['target', 'vpd_anom', 'soil_anom', 'lai_anom', 'memory', 'month', 'igbp']
    for col in expected_cols:
        assert col in df.columns

    # Check if the index is a MultiIndex with time, lat, lon
    assert isinstance(df.index, pd.MultiIndex)
    assert 'time' in df.index.names
    assert 'lat' in df.index.names
    assert 'lon' in df.index.names

def test_train_xgboost_and_export(fire_generator_instance, tmp_path):
    """Tests the training and export functionality."""
    # Create a simple dataframe for training
    df = pd.DataFrame({
        'vpd_anom': np.random.rand(100),
        'soil_anom': np.random.rand(100),
        'lai_anom': np.random.rand(100),
        'memory': np.random.rand(100),
        'month': np.random.randint(1, 13, 100),
        'igbp': np.random.randint(1, 17, 100),
        'target': np.random.rand(100)
    })

    model = fire_generator_instance.train_xgboost(df)
    assert model is not None

    # Test export
    output_file = tmp_path / "test_lut.bin"
    fire_generator_instance.export_binary_lut(model, filename=str(output_file))
    assert output_file.exists()
    assert output_file.stat().st_size > 0
