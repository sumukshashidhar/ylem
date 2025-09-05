"""Minimal Hugging Face-backed Cleaner.

Dead-simple wrapper around `transformers` so you can do:

    import ylem
    cleaner = ylem.Cleaner("nano", system_prompt="Strength: Medium")
    # or "large" or an HF repo id; you can also override aliases
    # like {"nano": "your/nano-model", "large": "your/large-model"}.
    markdown = cleaner("some text")

Adds:
 - Model selection via aliases ("nano"/"large") or direct HF repo id.
 - Optional `system_prompt` at init or per-call.
 - Suppressed Transformers warnings for a cleaner output.
 - `loguru`-based debug logs with generation timing and token stats.

Assumes dependencies are installed and available. Fails fast otherwise.
"""

from __future__ import annotations

from typing import Any, Literal
from time import perf_counter
import os
import warnings

from transformers import pipeline
from transformers.utils import logging as hf_logging
from loguru import logger


# Accepted model sizes for the initial API.
ModelSize = Literal["nano", "large"]


DEFAULT_MODELS: dict[str, str] = {
    # Small instruction-tuned Gemma 3 (works for quick local tests)
    "nano": "textcleanlm/gemma-3-270m-0.05-ckpt",
    # Placeholder: map 'large' to the same by default. Override via env
    # var YLEM_MODEL_LARGE or pass a HF repo id to Cleaner(model=...).
    "large": "textcleanlm/gemma-3-4b-0.01-ckpt",
}


class Cleaner:
    """Create a text cleaner that outputs Markdown.

    Parameters
    - model: alias ("nano"/"large") or HF repo id (e.g. "google/gemma-3-270m-it").
    - aliases: optional mapping to override default aliases, e.g.
      {"nano": "your/nano", "large": "your/large"}.
    - system_prompt: default system prompt used for all calls unless overridden.
    - suppress_warnings: if True, mute most Transformers/HF Hub warnings.
    """

    def __init__(
        self,
        model: ModelSize | str = "nano",
        *,
        aliases: dict[str, str] | None = None,
        system_prompt: str | None = "Strength: Medium",
        suppress_warnings: bool = True,
    ) -> None:
        # Resolve model id using provided alias map, env vars, or raw id.
        alias_map = {**DEFAULT_MODELS, **(aliases or {})}
        name = str(model)
        base_id = alias_map.get(name, name)
        self.model_id: str = (
            os.getenv(f"YLEM_MODEL_{name.upper()}", base_id)
            if name in alias_map
            else base_id
        )

        if suppress_warnings:
            # Quiet down Transformers + HF Hub advisory logs and general warnings.
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            hf_logging.set_verbosity_error()
            warnings.filterwarnings("ignore", module=r"transformers(\..*)?")
            warnings.filterwarnings("ignore", category=UserWarning)

        # Initialize pipeline
        self.pipe = pipeline("text-generation", model=self.model_id)
        tok = self.pipe.tokenizer
        # Avoid pad/eos warnings during generation
        try:
            if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
                tok.pad_token = tok.eos_token  # type: ignore[attr-defined]
        except Exception:
            # Best-effort; continue even if tokenizer does not expose these attrs.
            pass

        # Store defaults
        self.system_prompt = system_prompt

        # Log some init details
        try:
            logger.debug(
                "Cleaner initialized | model_id='{}' | device={} | tokenizer='{}'",
                self.model_id,
                getattr(self.pipe, "device", None),
                type(tok).__name__,
            )
        except Exception:
            pass

    def __call__(
        self,
        text: str,
        *,
        max_new_tokens: int = 2048,
        system_prompt: str | None = None,
        **gen_kwargs: Any,
    ) -> str:
        """Generate Markdown from input text using an HF model.

        Parameters
        - text: input text to clean.
        - max_new_tokens: generation budget.
        - system_prompt: optional override for the default system prompt.
        - gen_kwargs: forwarded to `transformers.pipeline(...)(...)`.
        """
        tok = self.pipe.tokenizer

        # Build messages and render chat template
        sys_msg = system_prompt if system_prompt is not None else self.system_prompt
        messages: list[dict[str, str]] = []
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        messages.append({"role": "user", "content": text})

        t_template_start = perf_counter()
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        t_template = perf_counter() - t_template_start

        # Tokenize (for stats only)
        t_encode_start = perf_counter()
        try:
            enc = tok(prompt, add_special_tokens=False)
            prompt_tokens = len(enc.get("input_ids", []))
        except Exception:
            prompt_tokens = 0
        t_encode = perf_counter() - t_encode_start

        # Generate
        t_gen_start = perf_counter()
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            **gen_kwargs,
        )
        t_gen = perf_counter() - t_gen_start

        generated = outputs[0]["generated_text"]

        # Tokenize output (for stats only)
        try:
            out_ids = tok(generated, add_special_tokens=False).get("input_ids", [])
            out_tokens = len(out_ids)
        except Exception:
            out_tokens = 0

        total_time = t_template + t_encode + t_gen
        toks_per_sec = (out_tokens / t_gen) if t_gen > 0 and out_tokens else 0.0

        logger.debug(
            (
                "gen stats | prompt_tokens={} | out_tokens={} | "
                "prep_time={:.3f}s (template={:.3f}s, encode={:.3f}s) | "
                "gen_time={:.3f}s | total={:.3f}s | tok/s={:.2f}"
            ),
            prompt_tokens,
            out_tokens,
            (t_template + t_encode),
            t_template,
            t_encode,
            t_gen,
            total_time,
            toks_per_sec,
        )

        return generated
