import xarray as xr
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from scipy.stats import binned_statistic_2d, mode

class UFSCATChemFireGenerator:
    def __init__(self, target_res=0.04):
        """ Initialize with a target resolution (0.04 deg ~ 4km). """
        self.res = target_res
        self.target_lats = np.arange(-90 + self.res/2, 90, self.res)
        self.target_lons = np.arange(-180 + self.res/2, 180, self.res)

        self.bins = {
            'vpd_anom':   np.linspace(-5, 15, 15),
            'soil_anom':  np.linspace(-0.5, 0.5, 15),
            'lai_anom':   np.linspace(-2, 2, 10),
            'memory':     np.linspace(0, 100, 10),
            'months':     np.arange(1, 13),
            'igbps':      np.arange(1, 18)
        }

    def aggregate_raw_data(self, ds_500m):
        """
        Aggregates a 2D spatial slice (i.e., a single time step) of 500m
        satellite data to the target grid using scipy.stats.binned_statistic_2d.
        """
        print("Starting Spatial Aggregation to 4km...")

        # Prepare bins and data for binned_statistic_2d
        lat_bins = np.arange(-90, 90 + self.res, self.res)
        lon_bins = np.arange(-180, 180 + self.res, self.res)

        lons_flat, lats_flat = np.meshgrid(ds_500m.lon.values, ds_500m.lat.values)
        frp_flat = ds_500m['FRP'].values.flatten()
        lai_flat = ds_500m['LAI'].values.flatten()
        igbp_flat = ds_500m['IGBP'].values.flatten()

        # Perform 2D binned statistics
        frp_sum_4km = binned_statistic_2d(lats_flat.flatten(), lons_flat.flatten(), frp_flat, statistic='sum', bins=[lat_bins, lon_bins]).statistic
        lai_4km = binned_statistic_2d(lats_flat.flatten(), lons_flat.flatten(), lai_flat, statistic='mean', bins=[lat_bins, lon_bins]).statistic
        igbp_4km = binned_statistic_2d(lats_flat.flatten(), lons_flat.flatten(), igbp_flat, statistic=(lambda x: mode(x, keepdims=True).mode.squeeze()), bins=[lat_bins, lon_bins]).statistic

        # Normalize FRP
        R = 6371.0
        dlon_rad = np.deg2rad(self.res)
        lat_rad = np.deg2rad(self.target_lats)
        lat_edges_top = lat_rad + np.deg2rad(self.res / 2)
        lat_edges_bottom = lat_rad - np.deg2rad(self.res / 2)
        target_pixel_area = (R**2) * dlon_rad * np.abs(np.sin(lat_edges_top) - np.sin(lat_edges_bottom))
        frp_4km = frp_sum_4km / target_pixel_area[:, np.newaxis]

        # Transpose the results to match (lat, lon) order and create the Dataset
        return xr.Dataset(
            {
                'FRP': (('lat', 'lon'), frp_4km),
                'LAI': (('lat', 'lon'), lai_4km),
                'IGBP': (('lat', 'lon'), igbp_4km)
            },
            coords={
                'lat': self.target_lats[:frp_4km.shape[0]],
                'lon': self.target_lons[:frp_4km.shape[1]]
            }
        )

    def generate_features(self, ds_4km, met_4km):
        """
        Calculates anomalies and fire memory (6-month lag).
        """
        print("Calculating Anomalies and Fire Memory...")
        # Align time
        common_time = np.intersect1d(ds_4km.time.values, met_4km.time.values)
        ds = ds_4km.sel(time=common_time)
        met = met_4km.sel(time=common_time)

        # 1. Targets and Weather Anomalies
        frp_climo = ds['FRP'].groupby('time.month').mean('time')
        target_ratio = ds['FRP'] / (frp_climo.sel(month=ds['time.month']) + 0.1)

        vpd_anom = met['vpd'] - met['vpd'].groupby('time.month').mean('time')
        soil_anom = met['soil_m'] - met['soil_m'].groupby('time.month').mean('time')
        lai_anom = ds['LAI'] - ds['LAI'].groupby('time.month').mean('time')

        # 2. Fire Memory (Rolling sum of FRP for fuel depletion)
        memory = ds['FRP'].rolling(time=6, min_periods=1).sum()

        # 3. Combine into a single dataset and convert to DataFrame
        ds_out = xr.Dataset({
            'target': target_ratio,
            'vpd_anom': vpd_anom,
            'soil_anom': soil_anom,
            'lai_anom': lai_anom,
            'memory': memory,
            'igbp': ds['IGBP']
        })

        df = ds_out.to_dataframe().dropna()
        # Explicitly add month from the 'time' index level
        df['month'] = df.index.get_level_values('time').month
        return df

    def train_xgboost(self, df):
        """
        Trains XGBoost with Monotonic Constraints for physical consistency.
        """
        print("Training XGBoost with Monotonic Constraints...")
        # Features: [vpd(+), soil(-), lai(+), memory(-), month(0), igbp(0)]
        features = ['vpd_anom', 'soil_anom', 'lai_anom', 'memory', 'month', 'igbp']
        monotone_constraint_values = (1, -1, 1, -1, 0, 0)

        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            monotone_constraints=str(monotone_constraint_values),
            tree_method="hist",
            reg_lambda=1.5,
            early_stopping_rounds=50
        )

        # Time-series split: use last 15% for early stopping
        split = int(len(df) * 0.85)
        X_train = df[features].iloc[:split]
        y_train = df['target'].iloc[:split]
        X_val = df[features].iloc[split:]
        y_val = df['target'].iloc[split:]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
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
                # Reshape predictions to match meshgrid order (vpd, soil, lai, memory),
                # then transpose to match LUT order (memory, lai, soil, vpd).
                preds_reshaped = preds.reshape(15, 15, 10, 10)
                lut[i_idx, m_idx, :, :, :, :] = np.transpose(preds_reshaped, (3, 2, 1, 0))

        lut.tofile(filename)
        print(f"Export Complete! Binary file saved to: {os.path.abspath(filename)}")
        print("Fortran Shape: (15, 15, 10, 10, 12, 17)")
