"""Application configuration.

single typed boundary between the process environment and the codebase.
Nothing else in this package reads os.environ.
"""

from functools import cache
from pathlib import Path
from typing import Literal 

#for learning: field for one-use attr. secretStr for obfuscating strings without needing to hash them
from pydantic import Field, SecretStr

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

    #what's this for?
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    #LLM Settings
    #currently hardcoded in
    llm_provider: Literal["google"] = "google"
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    
    #assign this when it is needed only
    google_api_key: SecretStr

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr

    #what's a property? and why do we need to know if is_dev?
    @property
    def is_dev(self) -> bool:
        return self.env == "dev"
    
@cache
def get_settings() -> Settings:
    "Build the settings once per process, called only on entry points"

    return Settings()


