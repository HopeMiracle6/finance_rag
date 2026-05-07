from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.utils import project_root, resolve_path


DEFAULT_CONFIG_PATH = "configs/rag_config.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = resolve_path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml(path)
    apply_env_overrides(config)
    return config


def apply_env_overrides(config: dict[str, Any]) -> None:
    if os.getenv("LLM_PROVIDER"):
        config.setdefault("llm", {})["provider"] = os.environ["LLM_PROVIDER"]
    if os.getenv("EMBEDDING_MODEL"):
        config.setdefault("embedding", {})["model_name"] = os.environ["EMBEDDING_MODEL"]
    if os.getenv("RERANKER_MODEL"):
        config.setdefault("reranker", {})["model_name"] = os.environ["RERANKER_MODEL"]
    if os.getenv("LLM_MODEL"):
        config.setdefault("llm", {})["model"] = os.environ["LLM_MODEL"]
    if os.getenv("QLORA_BASE_MODEL"):
        config.setdefault("llm", {})["base_model"] = os.environ["QLORA_BASE_MODEL"]
    if os.getenv("QLORA_ADAPTER_PATH"):
        config.setdefault("llm", {})["adapter_path"] = os.environ["QLORA_ADAPTER_PATH"]
    if os.getenv("QLORA_LOAD_IN_4BIT"):
        config.setdefault("llm", {})["load_in_4bit"] = os.environ["QLORA_LOAD_IN_4BIT"].lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }


def config_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return resolve_path(value, project_root())
