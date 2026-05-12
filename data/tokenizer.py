"""Tokenizer loading helpers for local text experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


PadTokenStrategy = Literal["existing", "eos", "bos", "unk", "new", "add_pad", "none", "error"]
PaddingSide = Literal["left", "right"]


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for loading a Hugging Face tokenizer lazily."""

    name_or_path: str | None = None
    tokenizer_name_or_path: str | None = None
    cache_dir: str | None = None
    revision: str | None = None
    use_fast: bool = True
    trust_remote_code: bool = False
    local_files_only: bool = False
    model_max_length: int | None = None
    padding_side: PaddingSide = "right"
    pad_token_strategy: PadTokenStrategy = "eos"
    new_pad_token: str = "[PAD]"

    def __post_init__(self) -> None:
        if (
            self.name_or_path is not None
            and self.tokenizer_name_or_path is not None
            and self.name_or_path != self.tokenizer_name_or_path
        ):
            raise ValueError("name_or_path and tokenizer_name_or_path must match when both are provided.")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'.")
        if self.pad_token_strategy not in {"existing", "eos", "bos", "unk", "new", "add_pad", "none", "error"}:
            raise ValueError(f"Unsupported pad_token_strategy {self.pad_token_strategy!r}.")
        if self.pad_token_strategy in {"new", "add_pad"} and not self.new_pad_token:
            raise ValueError("new_pad_token must be non-empty when pad_token_strategy='new'.")
        if self.model_max_length is not None and self.model_max_length <= 0:
            raise ValueError("model_max_length must be positive when provided.")

    @property
    def resolved_name_or_path(self) -> str | None:
        return self.name_or_path if self.name_or_path is not None else self.tokenizer_name_or_path


def load_tokenizer(config: TokenizerConfig) -> Any:
    """Load a tokenizer without importing ``transformers`` at module import time."""

    if not config.resolved_name_or_path:
        raise ValueError("tokenizer_name_or_path is required for real text/jsonl datasets.")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only without test monkeypatching.
        raise ImportError("load_tokenizer requires the optional 'transformers' package.") from exc

    kwargs: dict[str, Any] = {
        "use_fast": config.use_fast,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
    }
    if config.cache_dir is not None:
        kwargs["cache_dir"] = config.cache_dir
    if config.revision is not None:
        kwargs["revision"] = config.revision
    if config.model_max_length is not None:
        kwargs["model_max_length"] = config.model_max_length

    tokenizer = AutoTokenizer.from_pretrained(config.resolved_name_or_path, **kwargs)
    _configure_padding(tokenizer, config)
    return tokenizer


def _configure_padding(tokenizer: Any, config: TokenizerConfig) -> None:
    _set_attr_if_possible(tokenizer, "padding_side", config.padding_side)
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return

    strategy = config.pad_token_strategy
    if strategy == "none":
        return
    if strategy in {"existing", "error"}:
        raise ValueError(
            "Tokenizer has no pad_token_id. Choose pad_token_strategy='eos', "
            "'bos', 'unk', 'new'/'add_pad', or 'none'."
        )
    if strategy in {"new", "add_pad"}:
        if hasattr(tokenizer, "add_special_tokens"):
            tokenizer.add_special_tokens({"pad_token": config.new_pad_token})
        else:
            _set_attr_if_possible(tokenizer, "pad_token", config.new_pad_token)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "pad_token", None) is None:
            raise ValueError("Failed to add a new pad token to the tokenizer.")
        return

    token = getattr(tokenizer, f"{strategy}_token", None)
    token_id = getattr(tokenizer, f"{strategy}_token_id", None)
    if token is None and token_id is None:
        raise ValueError(f"Tokenizer has no {strategy}_token to reuse as padding.")
    if token is not None:
        _set_attr_if_possible(tokenizer, "pad_token", token)
    if token_id is not None:
        _set_attr_if_possible(tokenizer, "pad_token_id", token_id)


def _set_attr_if_possible(obj: Any, name: str, value: Any) -> None:
    try:
        setattr(obj, name, value)
    except AttributeError:
        pass
