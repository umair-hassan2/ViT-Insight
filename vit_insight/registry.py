from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import yaml


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    family: str
    objective: str
    prefix_tokens: int
    patch_size: int
    num_layers: int
    text_conditioned: bool = False
    # False for models with no CLS token (e.g. SigLIP); CLS-row methods are unavailable.
    cls_attention: bool = True


class Registry:
    def __init__(self, path: str | None = None):
        if path is None:
            text = resources.files("vit_insight.configs").joinpath("models.yaml").read_text()
        else:
            with open(path) as f:
                text = f.read()
        raw = yaml.safe_load(text).get("models", [])
        self._specs = {m["id"]: ModelSpec(**m) for m in raw}

    def ids(self) -> list[str]:
        return list(self._specs)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as e:
            raise KeyError(f"unknown model {model_id!r}; known: {self.ids()}") from e

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._specs

    def __iter__(self):
        return iter(self._specs.values())
