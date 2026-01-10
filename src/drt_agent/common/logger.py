from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class JsonlLogger:
    """超轻量 JSONL 日志：每行一个 dict，方便你后续画图。"""
    path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
