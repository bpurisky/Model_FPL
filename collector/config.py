"""Loads config/collector.yaml into typed config objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# pyyaml: not in the stack locked by fpl-trends-superprompt.md §1.1, added here
# because §1.2 mandates YAML config files (config/collector.yaml,
# config/scoring_*.yaml) and something has to parse them.
import yaml


@dataclass
class ApiConfig:
    base_url: str
    user_agent: str
    rate_limit_per_second: float
    max_retries: int
    backoff_base_seconds: float
    backoff_jitter_seconds: float
    deadline_blackout_minutes: int

    def client_kwargs(self) -> dict:
        return {
            "base_url": self.base_url,
            "user_agent": self.user_agent,
            "rate_limit_per_second": self.rate_limit_per_second,
            "max_retries": self.max_retries,
            "backoff_base": self.backoff_base_seconds,
            "backoff_jitter": self.backoff_jitter_seconds,
        }


@dataclass
class StorageConfig:
    raw_dir: str
    distilled_dir: str
    reference_dir: str
    raw_retention_days: int


@dataclass
class CollectorConfig:
    api: ApiConfig
    storage: StorageConfig
    own_entry_id: int | None


def load_config(path: Path) -> CollectorConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    api_raw = raw["api"]
    storage_raw = raw["storage"]
    return CollectorConfig(
        api=ApiConfig(
            base_url=api_raw["base_url"],
            user_agent=api_raw["user_agent"],
            rate_limit_per_second=float(api_raw["rate_limit_per_second"]),
            max_retries=int(api_raw["max_retries"]),
            backoff_base_seconds=float(api_raw["backoff_base_seconds"]),
            backoff_jitter_seconds=float(api_raw["backoff_jitter_seconds"]),
            deadline_blackout_minutes=int(api_raw["deadline_blackout_minutes"]),
        ),
        storage=StorageConfig(
            raw_dir=storage_raw["raw_dir"],
            distilled_dir=storage_raw["distilled_dir"],
            reference_dir=storage_raw["reference_dir"],
            raw_retention_days=int(storage_raw["raw_retention_days"]),
        ),
        own_entry_id=raw.get("own_entry_id"),
    )
