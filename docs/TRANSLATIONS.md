# Translation Contribution Guide

This project supports internationalization (i18n) using JSON translation files located in the `src/i18n/` directory.

## Adding a New Language

1. Create a new JSON file in `src/i18n/` using the language code as the filename.

Example:

```text
src/i18n/de.json
```

2. Copy the contents of `src/i18n/en.json` into the new file.

3. Replace the English text with translations while keeping every key unchanged.

4. Register the language in `src/i18n/translator.py` by adding it to the `_SUPPORTED_LANGUAGES` dictionary.

5. Restart the application so the new translation file is loaded.

Example:

```python
_SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}
```

## JSON Structure

Each translation file is a JSON object where every key maps to a translated string.

Example:

```json
{
  "title": "Semantic Plagiarism Detection System",
  "language": "Language",
  "theme": "Theme"
}
```

Strings may contain placeholders that are substituted at runtime.

Example:

```json
{
  "warn_ai_prob": "AI Prob: {doc_a}: {ai_a:.1%} | {doc_b}: {ai_b:.1%}"
}
```

Do not modify placeholder names or formatting.

## Key Naming Conventions

- Keep all keys identical across every language file.
- Never remove existing keys.
- Use lowercase snake_case naming.
- Only translate the values.
- Preserve placeholders such as `{count}`, `{threshold}`, and `{doc_a}` exactly as written.

## Validation

Before submitting a translation:

- Ensure the JSON file is valid.
- Confirm the new language file contains the same set of keys as `src/i18n/en.json`.
- Verify placeholder names remain unchanged.

Run the project's test suite before opening a pull request:

```bash
python scripts/run_tests.py --parallel --coverage
```
