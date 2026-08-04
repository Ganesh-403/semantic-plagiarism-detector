# Application Configuration Schema

## Overview

This document describes every configuration variable managed by
`src/core/app_config.py`.

Configurations are loaded from environment variables and fall back to
default values where applicable.

---

## Configuration Variables

| Config Key | Environment Variable | Type | Default | Required | Description |
|------------|----------------------|------|---------|----------|-------------|
| APP_TITLE | APP_TITLE | string | Semantic Plagiarism Detection System | No | Application title |

## Environment Variables

### APP_TITLE

- Type: string
- Default: Semantic Plagiarism Detection System
- Required: No

Controls the application title. Empty or whitespace-only values fall back to the default so a malformed environment variable cannot leave the browser or page title blank.

Example:

```bash
APP_TITLE="Custom Plagiarism Detector"
```

## Supported Types

- string
- integer
- float
- boolean
- list
- dictionary

## Default Behaviour

If an environment variable is missing, the application uses the default
defined in `app_config.py`.

Invalid values are rejected or replaced with the default according to the
validation rules implemented in the configuration loader.

## Example

```env
APP_TITLE="Custom Plagiarism Detector"
```
