# ylem

Minimal package with a small, friendly `Cleaner` API for turning text into Markdown using Hugging Face models.

Quick Start

```python
import ylem

# Use an alias ("nano"/"large") or a full HF repo id
cleaner = ylem.Cleaner("nano", system_prompt="Strength: Medium")

text = "some content from the internet"
markdown = cleaner(text)  # returns generated text
```

Options (Cleaner)
- model: alias ("nano"/"large") or HF repo id (e.g. `google/gemma-3-270m-it`).
- aliases: optional dict to override aliases in code, e.g. `{ "nano": "your/nano" }`.
- system_prompt: default system prompt; can be overridden per call.
- suppress_warnings: hides most Transformers/HF Hub warnings (default: True).
- repetition_penalty: slight penalty to reduce repeats (default: 1.05). Set higher (e.g. 1.1) for stronger effect.

Per-call Overrides

```python
out = cleaner(
    text,
    max_new_tokens=128,
    repetition_penalty=1.1,      # override instance default
    do_sample=True,
    temperature=0.7,             # any transformers generation kwargs
    system_prompt="Strength: High; Output Markdown only.",
)
```

Model Selection
- Built-in aliases map to Hugging Face IDs:
  - "nano" -> `google/gemma-3-270m-it`
  - "large" -> `google/gemma-3-270m-it` (same by default)
- Override via env: set `YLEM_MODEL_NANO` / `YLEM_MODEL_LARGE` to a different repo id.
- Override in code: `ylem.Cleaner("nano", aliases={"nano": "your-org/your-model"})`.

Logging & Stats
- Uses `loguru` for pretty debug logs.
- Logs include: prompt tokens, output tokens, prep time (template+encode), generation time, total, and tokens/sec, plus the effective repetition penalty.
- To reduce verbosity in your app:
  ```python
  from loguru import logger
  import sys
  logger.remove()
  logger.add(sys.stderr, level="INFO")  # or "WARNING"
  ```

Warnings
- By default, most Transformers/HF Hub warnings are suppressed inside `Cleaner`.
- Disable suppression if you want full logs: `ylem.Cleaner(..., suppress_warnings=False)`.

Notes
- Backend: Uses `transformers` text-generation pipeline. Install a backend like PyTorch.
- Chat template: Renders a simple system+user chat prompt using the tokenizer’s template.
- Defaults: `max_new_tokens=2048`, return only generated continuation (not prompt).
- Python: Requires Python 3.12+.
