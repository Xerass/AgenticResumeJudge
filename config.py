#create a settings class for easier API usage,
#we can simply inject settings as a dependency for Fast API
#validating our env vars on startup

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #default settings for now (we can later change this)
    llm_provider: str = "google"
    llm_model: str = "gemini-2.5-flash"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    #load sensitive info from .env vars automatically
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    #subclass config, will pull from env
    class Config:
        env_file = ".env"

#create a singleton settings instance that gets imported by all other modules
settings = Settings()