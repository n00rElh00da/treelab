from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = ""
    session_secret: str = "local-development-only-change-me"
    app_env: str = "development"
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

    @model_validator(mode="after")
    def validate_cookie_security(self) -> "Settings":
        if self.app_env.lower() in {"production", "prod"} and not self.cookie_secure:
            raise ValueError("COOKIE_SECURE must be true in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
