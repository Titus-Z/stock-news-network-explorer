# Security policy

## Credentials

Never place API keys in source files, notebooks, screenshots, or committed
configuration. Use `ALPHA_VANTAGE_API_KEY` and `OPENAI_API_KEY` environment
variables.

If a key is ever committed, removing the visible line is insufficient. Revoke
or rotate the key at the provider, then remove it from Git history before the
repository becomes public.

## Untrusted content

Downloaded news, user-entered text, cached CSV/JSON files, model responses, and
generated HTML must be treated as untrusted. Do not load data or model artifacts
from unknown sources.

Report a vulnerability privately to the repository owner rather than opening
an issue containing credentials or exploit details.
