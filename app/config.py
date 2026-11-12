import os
from pathlib import Path


def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


class Settings:
    """Read from the environment on every instantiation so tests can override cheaply."""

    def __init__(self):
        self.data_dir = Path(os.getenv("DATA_DIR", "./data_store"))
        self.max_upload_mb = _i("MAX_UPLOAD_MB", 25)
        self.max_files = _i("MAX_FILES", 10)
        self.max_rows = _i("MAX_ROWS", 1_000_000)
        self.llm_provider = os.getenv("LLM_PROVIDER", "none").lower()
        self.llm_model = os.getenv("LLM_MODEL", "")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.llm_timeout_s = _f("LLM_TIMEOUT_S", 20)
        self.llm_max_retries = _i("LLM_MAX_RETRIES", 2)
        self.llm_backoff_s = _f("LLM_BACKOFF_S", 0.5)
        self.query_timeout_s = _f("QUERY_TIMEOUT_S", 10)
        self.date_dayfirst = os.getenv("DATE_DAYFIRST", "false").lower() == "true"
        self.api_token = os.getenv("API_TOKEN", "")
        # hard limits enforced by the plan validator
        self.max_limit = 100
        self.preview_rows = 20
        self.max_joins = 3
        self.max_filters = 10
        self.max_group_by = 3
        self.max_metrics = 4
        self.max_in_values = 50


def get_settings() -> Settings:
    return Settings()
