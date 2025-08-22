from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    cors_origins: list[str] = ["http://localhost:5173"]
    ollama_url: str = "http://localhost:11434"  # where `ollama serve` runs
    generation_default: str = "llama3"

settings = Settings()
