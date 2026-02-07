from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    CSV_INPUT_PATH: str
    WEB_CSV_INPUT_PATH: str
    PDF_INPUT_PATH: str | None = None
    # Supabase (Postgres)
    SUPABASE_DB__HOST: str
    SUPABASE_DB__PORT: int = 5432
    SUPABASE_DB__NAME: str = "postgres"
    SUPABASE_DB__USER: str = "postgres"
    SUPABASE_DB__PASSWORD: str

    # Qdrant
    QDRANT__URL: str
    QDRANT__API_KEY: str
    QDRANT__COLLECTION: str = "pharma_suggestions"

    # OpenRouter
    OPENROUTER__API_KEY: str
    OPENROUTER__API_URL: str = "https://openrouter.ai/api/v1"

    # App
    LOG_LEVEL: str = "INFO"
    API_KEY: str | None = None
    API_RATE_LIMIT: str = "20/minute"
    ENTREZ_EMAIL: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

settings = Settings()
