import os
import pandas as pd
from sofiev_model.fortran_exporter import export_fortran_lut
from sofiev_model.sofiev_tuner import SofievTuner

def test_export_fortran_lut():
    """
    Tests that the export_fortran_lut function creates a file.
    """
    tuner = SofievTuner()
    # Create a dummy dataframe to fit the scaler and PCA
    df = pd.DataFrame({
        'frp_total': [100, 200, 300],
        'h_abl': [1000, 1500, 2000],
        'n_ft': [0.01, 0.012, 0.015],
        'wind_speed': [5, 10, 15],
        'h_obs_misr': [1200, 1800, 2500]
    })
    df = tuner.prepare_features(df)
    tuner.train(df)

    filename = "test_lut.txt"
    export_fortran_lut(tuner, filename)
    assert os.path.exists(filename)
    os.remove(filename)
