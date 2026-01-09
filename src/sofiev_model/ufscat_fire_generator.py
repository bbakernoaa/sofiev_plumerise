import os
import datetime
from typing import Dict, Tuple

import numpy as np
import xarray as xr
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter

# Import your rigorous vectorized routines
from .cffwi import FWI_Engine_Vectorized


class FireEmissionGenerator:
    """
    Generates daily fire emissions by scaling a climatology with an ML model.

    This class integrates a trained XGBoost model with the Canadian Forest
    Fire Weather Index (FWI) system to produce daily, high-resolution
    estimates of fire emissions suitable for atmospheric chemistry models.

    Parameters
    ----------
    model_path : str
        Path to the trained XGBoost model file (.json or .model).
    climo_path : str
        Path to the GBBEPx climatology NetCDF file. This file should contain
        a variable 'emissions' with dimensions (month, lat, lon).
    target_res : float, optional
        The target resolution in decimal degrees for the output grid.
        Default is 0.04, approximately 4km.

    Attributes
    ----------
    res : float
        Grid resolution in degrees.
    target_lats : np.ndarray
        Latitude coordinates for the target grid.
    target_lons : np.ndarray
        Longitude coordinates for the target grid.
    fwi_engine : FWI_Engine_Vectorized
        An instance of the vectorized FWI calculation engine.
    model : xgb.XGBRegressor
        The loaded XGBoost regressor model.
    climo : xr.Dataset
        The loaded GBBEPx climatology, opened with dask chunks for lazy loading.
    """
    def __init__(self, model_path: str, climo_path: str, target_res: float = 0.04):
        self.res = target_res
        self.target_lats = np.arange(-90 + self.res/2, 90, self.res)
        self.target_lons = np.arange(-180 + self.res/2, 180, self.res)

        # Initialize FWI Engine
        self.fwi_engine = FWI_Engine_Vectorized()

        # Load the XGBoost model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Load the GBBEPx Climatology lazily using dask
        self.climo = xr.open_dataset(climo_path, chunks="auto")

    def calculate_vpd(self, t2m: np.ndarray, rh2m: np.ndarray) -> np.ndarray:
        """
        Calculate Vapor Pressure Deficit (VPD).

        Uses the Tetens equation to derive saturated vapor pressure from
        2-meter temperature and relative humidity.

        Parameters
        ----------
        t2m : np.ndarray
            2-meter temperature in Kelvin.
        rh2m : np.ndarray
            2-meter relative humidity in percent.

        Returns
        -------
        np.ndarray
            Vapor Pressure Deficit in hPa.
        """
        # Tetens equation for saturated vapor pressure
        t_c = t2m - 273.15
        es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        return es * (1.0 - rh2m / 100.0)

    def get_fire_memory(self, history_ds: xr.Dataset, current_time: pd.Timestamp) -> np.ndarray:
        """
        Calculate a 6-month cumulative Fire Radiative Power (FRP) memory.

        This serves as a proxy for fuel depletion, reducing fire intensity in
        areas that have recently burned.

        Parameters
        ----------
        history_ds : xr.Dataset
            An xarray Dataset containing a time-series of previously
            generated 'FRP' emissions.
        current_time : pd.Timestamp
            The current timestamp for the simulation step.

        Returns
        -------
        np.ndarray
            A 2D numpy array of cumulative FRP over the last 6 months.
        """
        six_months_ago = current_time - pd.DateOffset(months=6)
        # Select and sum emissions history
        memory = history_ds['FRP'].sel(time=slice(six_months_ago, current_time)).sum(dim='time')
        return memory.values

    def run_step(
        self,
        ufs_met: xr.Dataset,
        prev_states: Dict[str, np.ndarray],
        memory_grid: np.ndarray,
        igbp_map: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Execute a single daily timestep for the global grid.

        This method orchestrates the calculation of FWI indices, the
        prediction from the XGBoost model, and the final scaling of the
        base emissions climatology.

        Parameters
        ----------
        ufs_met : xr.Dataset
            An xarray Dataset containing the current day's meteorological
            drivers (t2m, rh2m, u10, v10, precip).
        prev_states : dict
            A dictionary containing the previous day's FWI moisture codes,
            with keys 'ffmc', 'dmc', and 'dc'.
        memory_grid : np.ndarray
            A 2D array of the 6-month cumulative FRP (fire memory).
        igbp_map : np.ndarray
            A 2D array of IGBP land cover classes.

        Returns
        -------
        tuple[np.ndarray, dict]
            A tuple containing:
            - final_emissions (np.ndarray): The scaled emissions field.
            - new_states (dict): The updated FWI moisture codes for the next day.
        """
        # 1. Parse Time
        current_dt = pd.to_datetime(ufs_met.time.values.item())
        month = current_dt.month

        # 2. Update FWI Moisture Codes via ccfwi.py
        wind_speed = np.sqrt(ufs_met['u10'].squeeze().values**2 + ufs_met['v10'].squeeze().values**2)
        new_ffmc = self.fwi_engine.calculate_ffmc(
            ufs_met['t2m'].squeeze().values, ufs_met['rh2m'].squeeze().values,
            wind_speed, ufs_met['precip'].squeeze().values, prev_states['ffmc']
        )
        new_dmc = self.fwi_engine.calculate_dmc(
            ufs_met['t2m'].squeeze().values, ufs_met['rh2m'].squeeze().values,
            ufs_met['precip'].squeeze().values, prev_states['dmc'], month
        )
        new_dc = self.fwi_engine.calculate_dc(
            ufs_met['t2m'].squeeze().values, ufs_met['precip'].squeeze().values,
            prev_states['dc'], month
        )

        # 3. Calculate Behavioral Indices
        bui = self.fwi_engine.calculate_bui(new_dmc, new_dc)

        # 4. Supplemental Predictors
        vpd = self.calculate_vpd(ufs_met['t2m'].values, ufs_met['rh2m'].values)

        # 5. ML Scaling
        # Feature vector alignment matches model training order
        X = np.stack([
            new_dc.ravel(),
            bui.ravel(),
            wind_speed.ravel(),
            vpd.ravel(),
            memory_grid.ravel(),
            igbp_map.ravel()
        ], axis=1)

        # Predict scaling factor
        raw_scale = self.model.predict(X).reshape(new_dc.shape)

        # 6. Post-processing: Gaussian smoothing and clipping
        smooth_scale = gaussian_filter(raw_scale, sigma=1.0)
        smooth_scale = np.clip(smooth_scale, 0.01, 20.0)

        # 7. Apply to Base Climatology
        base_emissions = self.climo['emissions'].sel(month=month).squeeze().values
        final_emissions = base_emissions * smooth_scale

        new_states = {
            'ffmc': new_ffmc,
            'dmc': new_dmc,
            'dc': new_dc
        }

        return final_emissions, new_states

    def save_state(self, states: Dict[str, np.ndarray], filename: str) -> None:
        """
        Save FWI moisture codes to a NetCDF file for restart capability.

        Includes data provenance by writing a history attribute.

        Parameters
        ----------
        states : dict
            A dictionary of 2D numpy arrays representing the FWI moisture
            codes ('ffmc', 'dmc', 'dc').
        filename : str
            The path for the output NetCDF file.
        """
        ds = xr.Dataset(
            {
                'ffmc': (['lat', 'lon'], states['ffmc']),
                'dmc': (['lat', 'lon'], states['dmc']),
                'dc': (['lat', 'lon'], states['dc'])
            },
            coords={'lat': self.target_lats, 'lon': self.target_lons}
        )
        ds.attrs["history"] = f"Created by Aero on {datetime.datetime.utcnow()} UTC"
        ds.to_netcdf(filename)

    def load_state(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Load FWI moisture codes from a previous day's output file.

        Uses dask for lazy loading to handle large state files efficiently.

        Parameters
        ----------
        filename : str
            The path to the input NetCDF state file.

        Returns
        -------
        dict
            A dictionary of FWI moisture codes ('ffmc', 'dmc', 'dc').
        """
        with xr.open_dataset(filename, chunks="auto") as ds:
            return {
                'ffmc': ds['ffmc'].values,
                'dmc': ds['dmc'].values,
                'dc': ds['dc'].values
            }
