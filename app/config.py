from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    roboflow_api_key: str = ""
    roboflow_model_id: str = ""
    match_threshold: float = 0.72
    database_path: str = "familiar-frame.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
