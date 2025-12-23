import numpy as np
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class AIOpsModel:
    def __init__(self):
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        print(f"🔧 Training model version {self.version}...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        score = self.model.score(X_scaled, y)
        print(f"✅ Model trained! R² score: {score:.4f}")
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def save(self, path="model.pkl"):
        joblib.dump(self, path)
        print(f"💾 Model saved to {path}")

if __name__ == "__main__":
    print("="*60)
    print("🚀 AIOps Model Training")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(1000, 4) * 2 + 5
    y = X[:, 0] * 2 + X[:, 1] * 0.5 - X[:, 2] + np.random.randn(1000) * 0.1
    
    model = AIOpsModel()
    model.fit(X, y)
    model.save("model.pkl")
    
    import os, shutil
    if os.path.exists("../app"):
        shutil.copy("model.pkl", "../app/model.pkl")
        print("📋 Model copied to app/")
    
    print("="*60)
