import os
import numpy as np
import xarray as xr
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter

# Import your rigorous vectorized routines
from ccfwi import FWI_Engine_Vectorized

class FireEmissionGenerator:
    def __init__(self, model_path, climo_path, target_res=0.04):
        """
        UFS/CATChem Fire Generator for RISE.
        
        Args:
            model_path (str): Path to the trained XGBoost .json or .model file.
            climo_path (str): Path to the aggregated GBBEPx climatology (4km).
            target_res (float): Resolution in degrees (0.04 ~ 4km).
        """
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
        
        # Load the GBBEPx Climatology
        self.climo = xr.open_dataset(climo_path)
        
    def calculate_vpd(self, t2m, rh2m):
        """
        Calculate Vapor Pressure Deficit (hPa).
        t2m: Temperature in Kelvin.
        rh2m: Relative Humidity in %.
        """
        # Tetens equation for saturated vapor pressure
        t_c = t2m - 273.15
        es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        return es * (1.0 - rh2m / 100.0)

    def get_fire_memory(self, history_ds, current_time):
        """
        Calculates 6-month cumulative FRP to handle biomass depletion.
        history_ds: Dataset containing the time-series of previously generated emissions.
        """
        six_months_ago = pd.to_datetime(current_time) - pd.DateOffset(months=6)
        # Select and sum emissions history
        memory = history_ds['FRP'].sel(time=slice(six_months_ago, current_time)).sum(dim='time')
        return memory.values

    def run_step(self, ufs_met, prev_states, memory_grid, igbp_map):
        """
        Executes a single daily timestep for the global 4km grid.
        
        Args:
            ufs_met (xr.Dataset): Current UFS met (t2m, rh2m, u10, v10, precip).
            prev_states (dict): Dict of 2D arrays (ffmc, dmc, dc) from previous day.
            memory_grid (np.ndarray): 2D array of cumulative FRP (6-month lag).
            igbp_map (np.ndarray): 2D array of IGBP land cover classes.
            
        Returns:
            emissions (np.ndarray): Scaled emissions for CATChem.
            new_states (dict): Updated FWI codes for the next day.
        """
        # 1. Parse Time
        current_dt = pd.to_datetime(ufs_met.time.values[0])
        month = current_dt.month
        
        # 2. Update FWI Moisture Codes via ccfwi.py
        # FFMC (Fine Fuel), DMC (Duff), DC (Drought)
        new_ffmc = self.fwi_engine.calculate_ffmc(
            ufs_met['t2m'].values, ufs_met['rh2m'].values, 
            np.sqrt(ufs_met['u10'].values**2 + ufs_met['v10'].values**2), 
            ufs_met['precip'].values, prev_states['ffmc']
        )
        
        new_dmc = self.fwi_engine.calculate_dmc(
            ufs_met['t2m'].values, ufs_met['rh2m'].values, 
            ufs_met['precip'].values, prev_states['dmc'], month
        )
        
        new_dc = self.fwi_engine.calculate_dc(
            ufs_met['t2m'].values, ufs_met['precip'].values, 
            prev_states['dc'], month
        )

        # 3. Calculate Behavioral Indices
        isi = self.fwi_engine.calculate_isi(new_ffmc, np.sqrt(ufs_met['u10'].values**2 + ufs_met['v10'].values**2))
        bui = self.fwi_engine.calculate_bui(new_dmc, new_dc)
        
        # 4. Supplemental Predictors
        vpd = self.calculate_vpd(ufs_met['t2m'].values, ufs_met['rh2m'].values)
        wind = np.sqrt(ufs_met['u10'].values**2 + ufs_met['v10'].values**2)

        # 5. ML Scaling
        # Feature vector alignment: [DC, BUI, Wind, VPD, Memory, IGBP]
        # Reshape to (N_pixels, N_features)
        X = np.stack([
            new_dc.ravel(), 
            bui.ravel(), 
            wind.ravel(), 
            vpd.ravel(), 
            memory_grid.ravel(), 
            igbp_map.ravel()
        ], axis=1)
        
        # Predict scaling factor
        raw_scale = self.model.predict(X).reshape(new_dc.shape)
        
        # 6. Post-processing: Gaussian smoothing and clipping
        # Sigma=1.0 at 4km provides enough smoothing to prevent numerical shocks in solver
        smooth_scale = gaussian_filter(raw_scale, sigma=1.0)
        smooth_scale = np.clip(smooth_scale, 0.01, 20.0) 

        # 7. Apply to Base Climatology
        # Get the GBBEPx base for this month
        base_emissions = self.climo['emissions'].sel(month=month).values
        final_emissions = base_emissions * smooth_scale
        
        new_states = {
            'ffmc': new_ffmc,
            'dmc': new_dmc,
            'dc': new_dc
        }
        
        return final_emissions, new_states

    def save_state(self, states, filename):
        """Saves FWI moisture codes to NetCDF for restart capability."""
        ds = xr.Dataset({
            'ffmc': (['lat', 'lon'], states['ffmc']),
            'dmc': (['lat', 'lon'], states['dmc']),
            'dc': (['lat', 'lon'], states['dc'])
        }, coords={'lat': self.target_lats, 'lon': self.target_lons})
        ds.to_netcdf(filename)

    def load_state(self, filename):
        """Loads FWI moisture codes from a previous day's output."""
        ds = xr.open_dataset(filename)
        return {
            'ffmc': ds['ffmc'].values,
            'dmc': ds['dmc'].values,
            'dc': ds['dc'].values
        }

# --- END OF FILE ---
