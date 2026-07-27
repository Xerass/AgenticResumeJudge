"""Application configuration.

single typed boundary between the process environment and the codebase.
Nothing else in this package reads os.environ.
"""

from functools import cache
from pathlib import Path
from typing import Literal , Self

#for learning: field for variable metadata to validate on. secretStr for masking strings during display can still get via .get_secret_value()
from pydantic import Field, SecretStr, model_validator

from pydantic_settings import BaseSettings, SettingsConfigDict

#Settings class is a basemodel, but we use BaseSettings  to inform it of data sources in a pydantic fashion


#project root is located at parents[2], src/agenticresume/settings.py
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    "Validated, Immutable Runtime Configs"
    

    #defines the how's of loading
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    #app, the actual content of settings

    env: Literal["dev", "prod"] = "dev"

    #defines what we ignore when it comes to python logging, anything above info is listened to
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    #LLM Settings
    llm_provider: Literal["google", "openai", "anthropic", "groq"] = "google"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    #allow none, to enable providing only the used API key
    google_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    @model_validator(mode = "after")
    def _active_provider_has_key(self) -> Self:
        key = {
            "google": self.google_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "groq": self.groq_api_key,
        }[self.llm_provider]

        if key is None:
            raise ValueError(f"{self.llm_provider}_api_key is required when llm_provider={self.llm_provider!r}")

        return self
    
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr
    
@cache
def get_settings() -> Settings:
    "Build the settings once per process, called only on entry points"

    return Settings()  # pyright: ignore[reportCallIssue]  # fields come from env, not args


