import os
import numpy as np
import xarray as xr
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter

# Import your rigorous vectorized routines
from .cffwi import FWI_Engine_Vectorized

class FireEmissionGenerator:
    def __init__(self, model_path: str, climo_path: str, target_res: float = 0.04):
        """UFS/CATChem Fire Generator for RISE.

        This class orchestrates the calculation of biomass burning emissions by
        integrating meteorological data, a fuel climatology, and a trained
        machine learning model to produce daily scaling factors.

        Parameters
        ----------
        model_path : str
            Path to the trained XGBoost .json or .model file.
        climo_path : str
            Path to the aggregated GBBEPx climatology NetCDF file. This file
            is expected to contain a monthly emission climatology.
        target_res : float, optional
            Target resolution in degrees for the output grid, by default 0.04,
            approximating a 4km grid spacing.

        """
        self.res = target_res
        self.target_lats = np.arange(-90 + self.res / 2, 90, self.res)
        self.target_lons = np.arange(-180 + self.res / 2, 180, self.res)

        # Initialize FWI Engine
        self.fwi_engine = FWI_Engine_Vectorized()

        # Load the XGBoost model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Load the GBBEPx Climatology lazily with Dask
        self.climo = xr.open_dataset(climo_path, chunks={'lat': 512, 'lon': 512})
        
    def calculate_vpd(self, t2m: np.ndarray, rh2m: np.ndarray) -> np.ndarray:
        """Calculate Vapor Pressure Deficit (VPD).

        This method uses the Tetens equation to estimate saturated vapor
        pressure from temperature.

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

    def get_fire_memory(self, history_ds: xr.Dataset, current_time: pd.Timestamp) -> xr.DataArray:
        """Calculate 6-month cumulative FRP for biomass depletion.

        Parameters
        ----------
        history_ds : xr.Dataset
            Time-series of previously generated FRP emissions. Must have a
            'time' coordinate.
        current_time : pd.Timestamp
            The current timestamp for the model run.

        Returns
        -------
        xr.DataArray
            A 2D DataArray of cumulative FRP over the last 6 months.
        """
        six_months_ago = current_time - pd.DateOffset(months=6)
        # Select and sum emissions history
        memory = history_ds['FRP'].sel(time=slice(six_months_ago, current_time)).sum(dim='time')
        return memory

    def run_step(self, ufs_met: xr.Dataset, prev_states: xr.Dataset,
                 memory_grid: xr.DataArray, igbp_map: xr.DataArray) -> tuple[xr.DataArray, xr.Dataset]:
        """Execute a single daily timestep for the global 4km grid.

        This is the core operational method. It takes the latest meteorological
        data and the previous day's FWI state, computes the next state, and
        predicts an emissions scaling factor using the XGBoost model.

        Parameters
        ----------
        ufs_met : xr.Dataset
            Current UFS meteorology. Must contain `t2m`, `rh2m`, `u10`, `v10`,
            and `precip` as DataArrays.
        prev_states : xr.Dataset
            Dataset of 2D arrays for `ffmc`, `dmc`, and `dc` from the
            previous day.
        memory_grid : xr.DataArray
            2D array of cumulative FRP (6-month lag).
        igbp_map : xr.DataArray
            2D array of IGBP land cover classes.

        Returns
        -------
        tuple[xr.DataArray, xr.Dataset]
            - The final scaled emissions as a 2D DataArray, preserving
              coordinates and including a history attribute.
            - An updated xr.Dataset containing the new FWI moisture codes
              (`ffmc`, `dmc`, `dc`).
        """
        # 1. Parse Time
        current_dt = pd.to_datetime(ufs_met.time.values.item())
        month = current_dt.month

        # 2. Update FWI Moisture Codes via ccfwi.py
        wind_speed = np.sqrt(ufs_met['u10']**2 + ufs_met['v10']**2)
        new_ffmc = self.fwi_engine.calculate_ffmc(
            ufs_met['t2m'].values, ufs_met['rh2m'].values,
            wind_speed.values, ufs_met['precip'].values,
            prev_states['ffmc'].values
        )
        new_dmc = self.fwi_engine.calculate_dmc(
            ufs_met['t2m'].values, ufs_met['rh2m'].values,
            ufs_met['precip'].values, prev_states['dmc'].values, month
        )
        new_dc = self.fwi_engine.calculate_dc(
            ufs_met['t2m'].values, ufs_met['precip'].values,
            prev_states['dc'].values, month
        )

        # 3. Calculate Behavioral Indices
        bui = self.fwi_engine.calculate_bui(new_dmc, new_dc)

        # 4. Supplemental Predictors
        vpd = self.calculate_vpd(ufs_met['t2m'].values, ufs_met['rh2m'].values)

        # 5. ML Scaling
        # Feature vector alignment: [DC, BUI, Wind, VPD, Memory, IGBP]
        X = np.stack([
            new_dc.ravel(),
            bui.ravel(),
            wind_speed.values.ravel(),
            vpd.ravel(),
            memory_grid.values.ravel(),
            igbp_map.values.ravel()
        ], axis=1)

        # Predict scaling factor and reshape
        raw_scale = self.model.predict(X).reshape(new_dc.shape)

        # 6. Post-processing: Gaussian smoothing and clipping
        smooth_scale = gaussian_filter(raw_scale, sigma=1.0)
        smooth_scale = np.clip(smooth_scale, 0.01, 20.0)

        # 7. Apply to Base Climatology
        base_emissions = self.climo['emissions'].sel(month=month).values
        final_emissions_np = base_emissions * smooth_scale

        # 8. Format Output
        # Preserve coordinates and add history
        final_emissions = xr.DataArray(
            final_emissions_np,
            coords=ufs_met.coords,
            dims=ufs_met['t2m'].dims,
            name='emissions'
        )
        history_log = (
            f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"Fire emissions generated with UFSCATChemFireGenerator."
        )
        final_emissions.attrs['history'] = history_log

        new_states_ds = xr.Dataset({
            'ffmc': (('lat', 'lon'), new_ffmc),
            'dmc': (('lat', 'lon'), new_dmc),
            'dc': (('lat', 'lon'), new_dc)
        }, coords=ufs_met.coords)

        return final_emissions, new_states_ds

    def save_state(self, states: xr.Dataset, filename: str) -> None:
        """Saves FWI moisture codes to NetCDF for restart capability."""
        states.to_netcdf(filename)

    def load_state(self, filename: str) -> xr.Dataset:
        """Loads FWI moisture codes from a previous day's output."""
        return xr.open_dataset(filename)

# --- END OF FILE ---
