from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import Any

import os

from loguru import logger
from transformers import pipeline
from transformers.utils import logging as hf_logging


class Size(StrEnum):
    """Model size aliases."""
    nano = "nano"
    medium = "medium"
    large = "large"


class Strength(StrEnum):
    """Cleaning aggressiveness levels."""
    low = "low"
    medium = "medium"
    high = "high"


DEFAULT_MODELS: dict[str, str] = {
    "nano": "Qwen/Qwen3-0.6B",
    "medium": "Qwen/Qwen3-4B-Instruct-2507",
    "large": "Qwen/Qwen3-4B-Instruct-2507",
}


@dataclass(slots=True)
class Cleaner:
    """Minimal text cleaner that prompts an LLM to return Markdown.

    Fields:
      - model: alias ("nano"/"medium"/"large") or HF repo id.
      - strength: cleaning level ("low"/"medium"/"high").
      - aliases: optional alias→repo map merged with built-ins.
      - system_prompt: default system message; built from strength if None.
      - suppress_warnings: hide HF warnings.
      - repetition_penalty: generation penalty to reduce repeats.
    """
    model: Size | str = Size.nano
    strength: Strength | str = Strength.low
    aliases: dict[str, str] | None = None
    system_prompt: str | None = None
    suppress_warnings: bool = True
    repetition_penalty: float | None = 1.05

    model_id: str = field(init=False)
    pipe: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize pipeline/tokenizer, set defaults, and configure logging/warnings."""
        amap = DEFAULT_MODELS | (self.aliases or {})
        name = self.model.value if isinstance(self.model, Size) else str(self.model)
        base = amap.get(name, name)
        if name in amap: base = os.getenv(f"YLEM_MODEL_{name.upper()}", base)

        if self.suppress_warnings:
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            hf_logging.set_verbosity_error()

        self.model_id = base
        self.pipe = pipeline("text-generation", model=self.model_id)

        tok = self.pipe.tokenizer
        if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token = tok.eos_token

        if self.system_prompt is None: self.system_prompt = self._build_system_prompt(self.strength)

        logger.debug("Cleaner init | model_id={} | tokenizer={}", self.model_id, type(tok).__name__)

    def _resolve_model_id(self, model: Size | str | None, aliases: dict[str, str] | None) -> str:
        """Resolve a repo id from alias/env/overrides; returns the effective HF repo id."""
        amap = DEFAULT_MODELS | (self.aliases or {}) | (aliases or {})
        name = (model.value if isinstance(model, Size) else str(model)) if model is not None else (self.model.value if isinstance(self.model, Size) else str(self.model))
        base = amap.get(name, name)
        if name in amap: base = os.getenv(f"YLEM_MODEL_{name.upper()}", base)
        return base

    def _ensure_pipe(self, *, model: Size | str | None = None, aliases: dict[str, str] | None = None, suppress_warnings: bool | None = None) -> None:
        """Reload the pipeline if the resolved model changed; optionally update warnings behavior."""
        if suppress_warnings is not None: self.suppress_warnings = suppress_warnings
        if self.suppress_warnings:
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            hf_logging.set_verbosity_error()

        desired = self._resolve_model_id(model, aliases)
        if desired != self.model_id:
            self.model_id = desired
            self.pipe = pipeline("text-generation", model=self.model_id)
            tok = self.pipe.tokenizer
            if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token = tok.eos_token
            logger.debug("Cleaner reload | model_id={} | tokenizer={}", self.model_id, type(tok).__name__)

    def _to_strength(self, s: Strength | str | None) -> Strength:
        """Normalize input to a Strength enum; defaults to low."""
        return s if isinstance(s, Strength) else Strength((s or "low").lower())

    def _build_system_prompt(self, s: Strength | str) -> str:
        """Build the default system prompt string for a given cleaning strength."""
        lvl = self._to_strength(s).value.capitalize()
        return f"Remove irrelevant elements from the content, and convert to markdown. Cleaning Strength: {lvl}"

    def __call__(
        self,
        text: str,
        *,
        max_new_tokens: int = 2048,
        system_prompt: str | None = None,
        strength: Strength | str | None = None,
        model: Size | str | None = None,
        aliases: dict[str, str] | None = None,
        suppress_warnings: bool | None = None,
        repetition_penalty: float | None = None,
        **gen_kwargs: Any,
    ) -> str:
        """Clean text into Markdown using the configured LLM.

        Supports per-call overrides for model/aliases/warnings, strength/system prompt,
        and forwards any extra kwargs to `transformers.pipeline("text-generation")`.
        Returns the generated text only.
        """
        self._ensure_pipe(model=model, aliases=aliases, suppress_warnings=suppress_warnings)
        tok = self.pipe.tokenizer
        sp = system_prompt if system_prompt is not None else (self._build_system_prompt(strength) if strength is not None else self.system_prompt)
        messages = ([{"role": "system", "content": sp}] if sp else []) + [{"role": "user", "content": text}]

        t0 = perf_counter()
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tok(prompt, add_special_tokens=False).get("input_ids", []))
        t_prep = perf_counter() - t0

        rp = repetition_penalty if repetition_penalty is not None else self.repetition_penalty
        kwargs = {"max_new_tokens": max_new_tokens, "return_full_text": False} | gen_kwargs
        if rp is not None: kwargs["repetition_penalty"] = float(rp)

        t1 = perf_counter()
        out = self.pipe(prompt, **kwargs)
        t_gen = perf_counter() - t1

        gen = out[0]["generated_text"]
        out_tokens = len(tok(gen, add_special_tokens=False).get("input_ids", []))
        tps = (out_tokens / t_gen) if t_gen > 0 and out_tokens else 0.0

        logger.debug(
            "gen | prompt_tokens={} | out_tokens={} | prep={:.3f}s | gen={:.3f}s | tok/s={:.2f} | rep_penalty={}",
            prompt_tokens, out_tokens, t_prep, t_gen, tps, rp,
        )
        return gen