import os
import logging
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

load_dotenv()


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=False, extra="ignore"
    )

    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    tavily_api_key: Optional[str] = Field(None, alias="TAVILY_API_KEY")

    llm_model_name: str = Field("gpt-4o-mini", alias="MODEL_NAME")
    temperature: float = Field(0.1, alias="TEMPERATURE")
    max_tokens: int = Field(1000, alias="MAX_TOKENS")

    kg_confidence_threshold: float = Field(0.15, alias="KG_CONFIDENCE_THRESHOLD")
    self_consistency_samples: int = Field(3, alias="SELF_CONSISTENCY_SAMPLES")

    # Ensemble scoring weights
    rag_weight: float = Field(0.4, alias="RAG_WEIGHT")
    kg_weight: float = Field(0.4, alias="KG_WEIGHT")
    consistency_weight: float = Field(0.2, alias="CONSISTENCY_WEIGHT")

    # Confidence scoring thresholds
    high_confidence_threshold: float = Field(0.8, alias="HIGH_CONFIDENCE_THRESHOLD")
    medium_confidence_threshold: float = Field(0.5, alias="MEDIUM_CONFIDENCE_THRESHOLD")

    # Retry configuration
    max_retries: int = Field(2, alias="MAX_RETRIES")

    corpus_path: str = Field("data/corpus_europe", alias="CORPUS_PATH")

    # Logging configuration
    log_level: str = Field("DEBUG", alias="LOG_LEVEL")
    log_format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        alias="LOG_FORMAT",
    )
    log_file: str = Field("logs/agent.log", alias="LOG_FILE")


def setup_environment():
    """Load environment variables from .env file."""
    load_dotenv()


def get_config() -> Config:
    """Get configuration with proper error handling."""
    return Config()


config = get_config()


def setup_logging():
    """Setup centralized logging configuration."""
    from pathlib import Path

    # Create logs directory if it doesn't exist
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.log_format,
        handlers=[logging.StreamHandler()],
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module."""
    return logging.getLogger(name)
