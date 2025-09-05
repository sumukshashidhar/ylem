# ylem

Minimal package scaffold with a public Cleaner API.

Usage

```python
import ylem

cleaner = ylem.Cleaner("nano")  # or "large" or an HF repo id
text = "some content from the internet"
markdown = cleaner(text)  # returns generated text
```

Notes
- Backend: This uses `transformers` for a text-generation pipeline. Install a backend like PyTorch.
- Models: Aliases map to Hugging Face IDs:
  - "nano" -> `google/gemma-3-270m-it`
  - "large" -> `google/gemma-3-270m-it` (override via env `YLEM_MODEL_LARGE`)
- Env overrides: Set `YLEM_MODEL_NANO` or `YLEM_MODEL_LARGE` to point to a different repo id.
- Chat template: If the tokenizer provides one, it’s applied with a simple system+user message; otherwise the raw text is used directly.
- Default generation: `max_new_tokens=2048`, and we request only generated continuation (not prompt) from the pipeline.
- Python: Requires Python 3.12+.
