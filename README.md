## ylem

Turn messy text into Markdown with a tiny, friendly `Cleaner`.

### Install
```bash
uv pip install ylem
```

### Quick start
```python
import ylem

cleaner = ylem.Cleaner("nano")  # Qwen/Qwen3-0.6B
text = "some content from the internet"
markdown = cleaner(text)
```

### What you get
- **Default behavior**: remove irrelevant bits and convert to Markdown.
- **Strength**: `low` (default), `medium`, `high`.

### Tweak as needed
```python
# Choose strength or switch models per call
markdown = cleaner(text, strength="medium", max_new_tokens=256)

# Or set on init
cleaner = ylem.Cleaner(model="medium", strength="high")
```
