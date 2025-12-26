from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SofievTuner:
    """
    A class to tune the Sofiev plume rise model using a physics-guided
    machine learning approach.
    """
    def __init__(self):
        """
        Initializes the SofievTuner with physical constants and ML models.
        """
        # Physical constants (Sofiev 2012 / Li et al 2020)
        self.alpha = 0.15 # PBL scaling
        self.gamma = 0.50 # Buoyancy exponent
        self.delta = 0.20 # Stability damping
        self.P0 = 1.0     # Ref Power
        self.N0 = 0.01    # Ref Stability

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs PCA to identify fire regimes (Intensity vs Met).

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with fire event data.

        Returns
        -------
        pd.DataFrame
            The DataFrame with added PCA features.
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

    def calculate_beta_target(self, row: pd.Series) -> float:
        """
        INVERSE PHYSICS: Calculate the Beta required to match observation perfectly.
        H = alpha*Habl + Beta * BuoyancyTerm * StabilityTerm

        Parameters
        ----------
        row : pd.Series
            A row of the DataFrame representing a single fire event.

        Returns
        -------
        float
            The calculated beta value.
        """
        term_pbl = self.alpha * row['h_abl']

        # Prevent division by zero / overflow
        buoyancy = (row['frp_total'] / self.P0) ** self.gamma
        stability = np.exp(-self.delta * row['n_ft'] / self.N0)
        denom = buoyancy * stability

        if denom < 1e-5:
            denom = 1e-5

        beta = (row['h_obs_misr'] - term_pbl) / denom
        return beta

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trains the physics-guided machine learning model.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame with fire event data.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the trained model's predictions.
        """
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
