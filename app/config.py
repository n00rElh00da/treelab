from functools import lru_cache
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


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
    explainability_max_rows: int = 1000
    session_days: int = 14
    cors_origins: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    @model_validator(mode="after")
    def validate_cookie_security(self) -> "Settings":
        if self.app_env.lower() in {"production", "prod"}:
            if not self.cookie_secure:
                raise ValueError("COOKIE_SECURE must be true in production")
            if self.session_secret == "local-development-only-change-me" or len(self.session_secret) < 32:
                raise ValueError("SESSION_SECRET must be a unique value of at least 32 characters in production")
            if not self.database_url:
                raise ValueError("DATABASE_URL must be set in production")
            database_url = make_url(self.database_url)
            if database_url.get_backend_name() != "postgresql":
                raise ValueError("DATABASE_URL must use PostgreSQL in production")
            if database_url.query.get("sslmode") not in {"require", "verify-ca", "verify-full"}:
                raise ValueError("DATABASE_URL must require TLS in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
