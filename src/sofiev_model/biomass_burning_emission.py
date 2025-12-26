import xarray as xr
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from scipy.stats import mode

class UFSCATChemFireGenerator:
    def __init__(self, target_res=0.04):
        """
        Initialize with a target resolution (0.04 deg ~ 4km).
        """
        self.res = target_res
        self.target_lats = np.arange(-90 + self.res/2, 90, self.res)
        self.target_lons = np.arange(-180 + self.res/2, 180, self.res)

        # Define LUT bins (Discretization for Fortran)
        self.bins = {
            'vpd_anom':   np.linspace(-5, 15, 15),
            'soil_anom':  np.linspace(-0.5, 0.5, 15),
            'lai_anom':   np.linspace(-2, 2, 10),
            'memory':     np.linspace(0, 100, 10), # Normalized FRP sum
            'months':     np.arange(1, 13),
            'igbps':      np.arange(1, 18)
        }

    def aggregate_raw_data(self, ds_500m, coarsen_factor=8):
        """
        Aggregates 500m satellite data (FRP, LAI, IGBP) to a target grid using coarsen.
        The coarsen_factor of 8 assumes a 500m source resolution and a 4km target (4000/500 = 8).
        """
        print("Starting Spatial Aggregation...")

        # Coarsen the dataset
        ds_coarse = ds_500m.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary='trim')

        # Define the aggregations
        frp_agg = ds_coarse.sum()['FRP']
        lai_agg = ds_coarse.mean()['LAI']

        # For IGBP, we use a custom function with reduce
        def robust_mode(arr, axis):
            mode_result, _ = mode(arr, axis=axis, nan_policy='omit')
            return mode_result

        igbp_agg = ds_coarse.reduce(robust_mode)['IGBP']

        return xr.Dataset({'FRP': frp_agg, 'LAI': lai_agg, 'IGBP': igbp_agg})

    def generate_features(self, ds_4km, met_4km):
        """
        Calculates anomalies and fire memory (6-month lag).
        """
        print("Calculating Anomalies and Fire Memory...")
        # Align time
        common_time = np.intersect1d(ds_4km.time, met_4km.time)
        ds = ds_4km.sel(time=common_time)
        met = met_4km.sel(time=common_time)

        # 1. Targets and Weather Anomalies
        frp_climo = ds['FRP'].groupby('time.month').mean()
        target_ratio = ds['FRP'] / (frp_climo.sel(month=ds['time.month']) + 0.1)

        vpd_anom = met['vpd'] - met['vpd'].groupby('time.month').mean()
        soil_anom = met['soil_m'] - met['soil_m'].groupby('time.month').mean()
        lai_anom = ds['LAI'] - ds['LAI'].groupby('time.month').mean()

        # 2. Fire Memory (Rolling sum of FRP for fuel depletion)
        memory = ds['FRP'].rolling(time=6, min_periods=1).sum()

        # 3. Flatten to DataFrame
        df = pd.DataFrame({
            'target': target_ratio.values.flatten(),
            'vpd_anom': vpd_anom.values.flatten(),
            'soil_anom': soil_anom.values.flatten(),
            'lai_anom': lai_anom.values.flatten(),
            'memory': memory.values.flatten(),
            'month': np.repeat(ds['time.month'].values, ds.lat.size * ds.lon.size),
            'igbp': np.tile(ds['IGBP'].values.flatten(), len(ds.time))
        }).dropna()

        return df

    def train_xgboost(self, df):
        """
        Trains XGBoost with Monotonic Constraints for physical consistency.
        """
        print("Training XGBoost with Monotonic Constraints...")
        # Features: [vpd(+), soil(-), lai(+), memory(-), month(0), igbp(0)]
        features = ['vpd_anom', 'soil_anom', 'lai_anom', 'memory', 'month', 'igbp']
        constraints = (1, -1, 1, -1, 0, 0)

        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            monotone_constraints=str(constraints),
            tree_method="hist",
            reg_lambda=1.5
        )

        # Time-series split: use last 15% for early stopping
        split = int(len(df) * 0.85)
        model.fit(
            df[features].iloc[:split], df['target'].iloc[:split],
            eval_set=[(df[features].iloc[split:], df['target'].iloc[split:])],
            early_stopping_rounds=50,
            verbose=100
        )
        return model

    def export_binary_lut(self, model, filename="fire_scaling_lut.bin"):
        """
        Predicts across all bins to create a 5D LUT and saves as Fortran-readable binary.
        """
        print("Building 5D Look-Up Table (this may take a few minutes)...")
        # LUT Shape: [IGBP, Month, Memory, LAI, Soil, VPD]
        shape = (17, 12, 10, 10, 15, 15)
        lut = np.zeros(shape, dtype=np.float32)

        for i_idx, i_val in enumerate(self.bins['igbps']):
            for m_idx, m_val in enumerate(self.bins['months']):
                # Create grid for a single IGBP/Month slice
                gv, gs, gl, gm = np.meshgrid(
                    self.bins['vpd_anom'], self.bins['soil_anom'],
                    self.bins['lai_anom'], self.bins['memory'],
                    indexing='ij'
                )

                batch = pd.DataFrame({
                    'vpd_anom': gv.ravel(),
                    'soil_anom': gs.ravel(),
                    'lai_anom': gl.ravel(),
                    'memory': gm.ravel(),
                    'month': m_val,
                    'igbp': i_val
                })

                preds = model.predict(batch[['vpd_anom', 'soil_anom', 'lai_anom', 'memory', 'month', 'igbp']])
                lut[i_idx, m_idx, :, :, :, :] = preds.reshape(10, 10, 15, 15)

        lut.tofile(filename)
        print(f"Export Complete! Binary file saved to: {os.path.abspath(filename)}")
        print(f"Fortran Shape: (15, 15, 10, 10, 12, 17)")

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    # 1. Initialize
    fire_sys = UFSCATChemFireGenerator(target_res=0.04)

    # 2. Load Datasets (Replace with your actual paths/file lists)
    # ds_500m = xr.open_mfdataset('path_to_gbbepx_history/*.nc')
    # met_4km = xr.open_mfdataset('path_to_ufs_meteorology/*.nc')

    # 3. Run Pipeline
    # ds_4km = fire_sys.aggregate_raw_data(ds_500m)
    # training_df = fire_sys.generate_features(ds_4km, met_4km)
    # xgb_model = fire_sys.train_xgboost(training_df)
    # fire_sys.export_binary_lut(xgb_model)
