"""Configuration and settings for Finie."""

import os
from pathlib import Path
from typing import Dict, Any

import yaml
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Settings(BaseSettings):
    """Application settings loaded from environment and config."""
    
    # API Keys
    openai_api_key: str = ""
    alpha_vantage_api_key: str = ""
    news_api_key: str = ""
    finnhub_api_key: str = ""
    
    # Paths
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    chromadb_dir: Path = PROJECT_ROOT / "data" / "chromadb"
    log_dir: Path = PROJECT_ROOT / "logs"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# Global settings and config
settings = Settings()
config = load_config()

# Ensure directories exist
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.chromadb_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
