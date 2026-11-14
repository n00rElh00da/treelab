from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = ""
    session_secret: str = "local-development-only-change-me"
    cookie_name: str = "webdev_app_session"
    cookie_secure: bool = False
    cookie_samesite: str = "lax"
    port: int = 3000
    max_upload_bytes: int = 25 * 1024 * 1024
    max_upload_rows: int = 100_000
    max_upload_columns: int = 200
    session_days: int = 14
    cors_origins: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
