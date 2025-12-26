Here is the complete, integrated Python script. This file contains the AWS GFS ingestion, Satellite data simulation, PCA/ML tuning engine, and the Fortran LUT exporter.
You will need the following dependencies:
pip install numpy pandas xarray s3fs cfgrib scikit-learn scipy matplotlib boto3
sofiev_smart_tuner.py
import os
import numpy as np
import pandas as pd
import xarray as xr
import s3fs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ==============================================================================
# SECTION 1: GFS ANALYSIS INGESTOR (AWS S3)
# ==============================================================================
class GFSIngestor:
    """
    Connects to the NOAA GFS bucket on AWS to retrieve analysis data (f000)
    for specific locations and times.
    """
    def __init__(self):
        # Connect to NOAA's Public S3 Bucket (Anonymous access)
        self.fs = s3fs.S3FileSystem(anon=True)
        self.bucket = "noaa-gfs-bdp-pds"
        print("Connected to AWS S3 (noaa-gfs-bdp-pds).")

    def get_analysis_point(self, target_time, lat, lon):
        """
        Fetches PBL Height (Surface) and Stability/Wind (Isobaric) 
        from GFS Analysis for a single point.
        """
        # 1. Round time to nearest 6H GFS cycle (00, 06, 12, 18)
        # GFS directories are organized by cycle start time
        hour = target_time.hour
        cycle_hour = (hour // 6) * 6
        cycle_dt = target_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        
        date_str = cycle_dt.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"
        
        # Path: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.f000
        s3_path = f"{self.bucket}/gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f000"
        
        # Adjust longitude (GFS uses 0-360)
        lon_gfs = lon if lon >= 0 else lon + 360
        
        try:
            # We use a context manager to ensure files close properly
            # NOTE: We use 'cfgrib' engine. Ensure ecCodes is installed on your system.
            print(f"   Fetching GFS from S3: {s3_path}...")
            
            # Create a mapper to stream only required bytes
            mapper = s3fs.S3Map(root=s3_path, s3=self.fs, check=False)

            # --- A. Surface Data (PBL Height) ---
            # Filter for surface level variables to speed up read
            ds_surf = xr.open_dataset(mapper, engine='cfgrib', 
                                     backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'surface'}})
            
            # GFS Variable 'hpbl' = Planetary Boundary Layer Height
            h_abl = ds_surf['hpbl'].sel(latitude=lat, longitude=lon_gfs, method='nearest').values.item()
            
            # --- B. Isobaric Data (Wind & Stability in Free Troposphere) ---
            # We need 850mb and 700mb for stability calc
            ds_iso = xr.open_dataset(mapper, engine='cfgrib', 
                                    backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa'}})
            
            # Extract Winds at 850mb (Transport Level)
            u850 = ds_iso['u'].sel(isobaricInhPa=850, latitude=lat, longitude=lon_gfs, method='nearest').values.item()
            v850 = ds_iso['v'].sel(isobaricInhPa=850, latitude=lat, longitude=lon_gfs, method='nearest').values.item()
            wind_speed = np.sqrt(u850**2 + v850**2)
            
            # Extract Temps for Brunt-Vaisala (N)
            t850 = ds_iso['t'].sel(isobaricInhPa=850, latitude=lat, longitude=lon_gfs, method='nearest').values.item()
            t700 = ds_iso['t'].sel(isobaricInhPa=700, latitude=lat, longitude=lon_gfs, method='nearest').values.item()
            
            # Calculate Potential Temperature (Theta)
            theta850 = t850 * (1000/850)**0.286
            theta700 = t700 * (1000/700)**0.286
            
            # Calc N (Stability)
            g = 9.81
            dz_approx = 1500.0 # Approx meters between 850mb and 700mb
            theta_avg = (theta850 + theta700) / 2.0
            d_theta = theta700 - theta850
            
            if d_theta > 0:
                n_ft = np.sqrt((g / theta_avg) * (d_theta / dz_approx))
            else:
                n_ft = 0.001 # Unstable/Neutral
                
            return h_abl, wind_speed, n_ft

        except Exception as e:
            print(f"   [Error] GFS Fetch Failed for {target_time}: {e}")
            # Return NaNs so we can drop them later
            return np.nan, np.nan, np.nan

# ==============================================================================
# SECTION 2: SATELLITE DATA SIMULATOR (MISR/TROPOMI)
# ==============================================================================
class SatelliteIngestor:
    """
    Generates synthetic 'Truth' data for demonstration.
    Replace 'simulate_data' with actual NetCDF readers for production.
    """
    def __init__(self, n_samples=200):
        self.n_samples = n_samples
    
    def get_collocated_dataset(self, gfs_ingestor):
        print("\n--- Generating/Ingesting Satellite Data ---")
        
        # 1. Create Synthetic Fire Events
        # Fires usually happen in afternoon, summer
        base_time = datetime(2023, 7, 15, 12, 0)
        times = [base_time + timedelta(hours=np.random.randint(0, 48)) for _ in range(self.n_samples)]
        lats = np.random.uniform(35, 48, self.n_samples) # US/Canada latitudes
        lons = np.random.uniform(-120, -100, self.n_samples)
        
        # 2. Fire Intensity (FRP) - Exponential distribution
        frp = np.random.exponential(500, self.n_samples) + 50
        
        # 3. Fetch GFS Data (Real AWS calls or Simulation for speed)
        # NOTE: For this demo, we will SIMULATE the GFS return to avoid 
        # hitting AWS 200 times and waiting 10 mins. 
        # UNCOMMENT the loop below to use real AWS data.
        
        h_abl_list = []
        n_ft_list = []
        wind_list = []
        
        print("   (Simulating GFS values for speed...)")
        h_abl_list = np.random.normal(1500, 400, self.n_samples)
        n_ft_list = np.abs(np.random.normal(0.012, 0.003, self.n_samples))
        wind_list = np.random.weibull(2, self.n_samples) * 6
        
        # --- REAL AWS FETCH LOOP (Uncomment for production) ---
        # for t, lat, lon in zip(times, lats, lons):
        #     h, w, n = gfs_ingestor.get_analysis_point(t, lat, lon)
        #     h_abl_list.append(h)
        #     n_ft_list.append(n)
        #     wind_list.append(w)
            
        # 4. Generate "Observed" Plume Heights (The Truth)
        # We assume truth follows a complex physics law we want to rediscover
        # Real Obs = Buoyancy - WindShear + RandomNoise
        h_obs = 200 * (frp**0.4) * np.exp(-0.5 * np.array(n_ft_list)/0.01) - (50 * np.array(wind_list)) + np.random.normal(0, 300, self.n_samples)
        h_obs = np.maximum(h_obs, np.array(h_abl_list) + 100) # Ensure it rises at least near PBL
        
        df = pd.DataFrame({
            'time': times,
            'lat': lats,
            'lon': lons,
            'frp_total': frp,
            'h_abl': h_abl_list,
            'n_ft': n_ft_list,
            'wind_speed': wind_list,
            'h_obs_misr': h_obs
        })
        
        return df.dropna()

# ==============================================================================
# SECTION 3: PHYSICS-GUIDED ML TUNER
# ==============================================================================
class SofievTuner:
    def __init__(self):
        # Physical constants (Sofiev 2012 / Li et al 2020)
        self.alpha = 0.15 # PBL scaling
        self.gamma = 0.50 # Buoyancy exponent
        self.delta = 0.20 # Stability damping
        self.P0 = 1.0     # Ref Power
        self.N0 = 0.01    # Ref Stability
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)

    def prepare_features(self, df):
        """
        Runs PCA to identify fire regimes (Intensity vs Met).
        """
        print("\n--- Running PCA Feature Engineering ---")
        data = df.copy()
        
        # Log transform FRP (Orders of magnitude difference)
        data['log_frp'] = np.log10(data['frp_total'])
        
        # Features for Regime Identification
        X_raw = data[['log_frp', 'h_abl', 'n_ft', 'wind_speed']].values
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Run PCA
        pca_features = self.pca.fit_transform(X_scaled)
        data['PC1'] = pca_features[:, 0]
        data['PC2'] = pca_features[:, 1]
        
        print(f"   PCA Explained Variance: {self.pca.explained_variance_ratio_}")
        return data

    def calculate_beta_target(self, row):
        """
        INVERSE PHYSICS: Calculate the Beta required to match observation perfectly.
        H = alpha*Habl + Beta * BuoyancyTerm * StabilityTerm
        """
        term_pbl = self.alpha * row['h_abl']
        
        # Prevent division by zero / overflow
        buoyancy = (row['frp_total'] / self.P0) ** self.gamma
        stability = np.exp(-self.delta * row['n_ft'] / self.N0)
        denom = buoyancy * stability
        
        if denom < 1e-5: denom = 1e-5
        
        beta = (row['h_obs_misr'] - term_pbl) / denom
        return beta

    def train(self, df):
        print("\n--- Training Physics-Guided ML ---")
        
        # 1. Calculate Targets
        df['target_beta'] = df.apply(self.calculate_beta_target, axis=1)
        
        # 2. Filter Unphysical Betas (Noise in data can cause Beta < 0)
        df_clean = df[(df['target_beta'] > 10) & (df['target_beta'] < 800)].copy()
        print(f"   Training on {len(df_clean)} valid fire events (filtered outliers).")
        
        # 3. Train Random Forest
        # Inputs: Physical Drivers + PCA Regime Context
        features = ['log_frp', 'wind_speed', 'h_abl', 'n_ft', 'PC1', 'PC2']
        
        X = df_clean[features]
        y = df_clean['target_beta']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.rf_model.fit(X_train, y_train)
        
        print(f"   Model R2 Score: {self.rf_model.score(X_test, y_test):.3f}")
        return df_clean

    def export_fortran_lut(self, filename="sofiev_smart_lut.txt"):
        """
        Generates a 2D Lookup Table (FRP vs Wind) for Fortran usage.
        Assumes mean H_abl and N_ft for the static table.
        """
        print(f"\n--- Exporting Fortran LUT to {filename} ---")
        
        # LUT Axes
        frp_axis = np.linspace(50, 5000, 50)
        wind_axis = np.linspace(0, 25, 25)
        
        # Background Mean Conditions (Static assumptions for 2D LUT)
        # In a fully dynamic system, you might pass these into the RF at runtime in Python
        mean_habl = 1500.0
        mean_nft = 0.012
        
        with open(filename, 'w') as f:
            f.write("! Smart Sofiev Tuning Table (Generated by ML)\n")
            f.write(f"! Axis 1: FRP (MW) [{len(frp_axis)} bins]\n")
            f.write(f"! Axis 2: Wind (m/s) [{len(wind_axis)} bins]\n")
            f.write("! Columns: FRP  WIND  BETA\n")
            
            for frp in frp_axis:
                for w in wind_axis:
                    # Construct Input Vector
                    # Note: We need to project this synthetic point into PCA space
                    # to get PC1/PC2.
                    raw_vec = np.array([[np.log10(frp), mean_habl, mean_nft, w]])
                    scaled_vec = self.scaler.transform(raw_vec)
                    pca_vec = self.pca.transform(scaled_vec)
                    
                    input_df = pd.DataFrame([{
                        'log_frp': np.log10(frp),
                        'wind_speed': w,
                        'h_abl': mean_habl,
                        'n_ft': mean_nft,
                        'PC1': pca_vec[0,0],
                        'PC2': pca_vec[0,1]
                    }])
                    
                    pred_beta = self.rf_model.predict(input_df)[0]
                    f.write(f"{frp:.1f}  {w:.1f}  {pred_beta:.4f}\n")
        
        print("   Export Complete.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Setup AWS GFS Connection
    gfs_handler = GFSIngestor()
    
    # 2. Ingest Data (Simulated Satellite + "Real" GFS structure)
    sat_handler = SatelliteIngestor(n_samples=500)
    df_raw = sat_handler.get_collocated_dataset(gfs_handler)
    
    # 3. Initialize Tuner
    tuner = SofievTuner()
    
    # 4. Feature Engineering (PCA)
    df_proc = tuner.prepare_features(df_raw)
    
    # 5. Train ML Model
    df_result = tuner.train(df_proc)
    
    # 6. Generate Lookup Table for Fortran
    tuner.export_fortran_lut()
    
    # 7. Validation Plot
    plt.figure(figsize=(10,6))
    plt.scatter(df_result['frp_total'], df_result['target_beta'], c=df_result['wind_speed'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='Wind Speed (m/s)')
    plt.xscale('log')
    plt.xlabel('Fire Radiative Power (MW)')
    plt.ylabel('Required Beta Parameter')
    plt.title('Tuned Physics Parameter (Beta) by Fire Intensity & Wind')
    plt.grid(True, alpha=0.3)
    plt.show()

