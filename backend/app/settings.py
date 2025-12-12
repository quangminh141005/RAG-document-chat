from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-small"
    CHROMA_DIR: str = "app/storage/chroma"
    COLLECTION: str = "docs"
    TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()