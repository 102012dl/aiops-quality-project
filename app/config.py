from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "AIOps Inference Service"
    version: str = "1.0.0"
    model_path: str = "model.pkl"
    drift_threshold: float = 3.0
    log_level: str = "INFO"
    gitlab_webhook: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
