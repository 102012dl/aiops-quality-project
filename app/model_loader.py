from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

class AIOpsModel:
    def __init__(self):
        self.version = "1.0.0"
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
