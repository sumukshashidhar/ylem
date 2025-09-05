"""Minimal Hugging Face-backed Cleaner.

Dead-simple wrapper around `transformers` so you can do:

    import ylem
    cleaner = ylem.Cleaner("nano")  # or "large" or an HF repo id
    markdown = cleaner("some text")

Assumes dependencies are installed and available. Fails fast otherwise.
"""

from __future__ import annotations

from typing import Any, Literal
from transformers import pipeline


# Accepted model sizes for the initial API.
ModelSize = Literal["nano", "large"]


DEFAULT_MODELS: dict[str, str] = {
    # Small instruction-tuned Gemma 3 (works for quick local tests)
    "nano": "google/gemma-3-270m-it",
    # Placeholder: map 'large' to the same by default. Override via env
    # var YLEM_MODEL_LARGE or pass a HF repo id to Cleaner(model=...).
    "large": "google/gemma-3-270m-it",
}


class Cleaner:
    """Create a text cleaner that outputs Markdown."""

    def __init__(self, model: ModelSize | str = "nano") -> None:
        import os

        name = str(model)
        base_id = DEFAULT_MODELS.get(name, name)
        self.model_id: str = (
            os.getenv(f"YLEM_MODEL_{name.upper()}", base_id)
            if name in DEFAULT_MODELS
            else base_id
        )
        self.pipe = pipeline("text-generation", model=self.model_id)

    def __call__(self, text: str, *, max_new_tokens: int = 2048, **gen_kwargs: Any) -> str:
        """Generate Markdown from input text using an HF model."""
        tok = self.pipe.tokenizer
        messages: list[dict[str, str]] = [
            {"role": "system", "content": "Strength: Medium"},
            {"role": "user", "content": text},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            **gen_kwargs,
        )
        return outputs[0]["generated_text"]
