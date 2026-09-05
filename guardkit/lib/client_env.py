"""ONE RULE for the key and the address every OpenAI-compatible seat client uses.

Three clients in this repo talk to an OpenAI-compatible endpoint — the stamp
normalizer's model fallback (``guardkit/orchestrator/stamp_model_fallback.py``),
the QAV shadow (``guardkit/qa/qav_shadow.py``) and the code-review seat
(``guardkit/qa/review_seat.py``). Each used to send the literal placeholder key
``not-needed``, and two of them had the address ``http://localhost:9000/v1``
written into the code. That was fine while every call went straight to
llama-swap, which ignores the key. Since 2026-09-03 the factory's calls go
through LiteLLM instead, which checks the key and answers a real 401 — a
placeholder key stopped a planning run on 2026-09-04. This module is the one
place that decides what key is sent and which address is called, so the three
clients cannot drift apart again.

**The key.** ``OPENAI_API_KEY`` when it is set and not blank; otherwise the
client's own existing placeholder. A machine without the variable therefore
behaves exactly as it did before. The value is returned and used — it is never
logged, printed, or put in an error message.

**The address, in order of precedence.** An explicit setting for that one
client wins (the QAV shadow's ``endpoint`` in its config block, the base URL a
caller hands the review seat); then that client's own environment variable
(``GUARDKIT_STAMP_MODEL_URL``, ``GUARDKIT_QAV_SHADOW_URL``,
``GUARDKIT_REVIEW_SEAT_URL``); then the shared ``OPENAI_BASE_URL``; then the
client's built-in default, which for the QAV shadow and the review seat is
``http://localhost:9000/v1`` and for the stamp fallback is deliberately
nothing at all (no endpoint configured means the model is never asked).
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

__all__ = [
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "PLACEHOLDER_API_KEY",
    "DEFAULT_BASE_URL",
    "resolve_api_key",
    "resolve_base_url",
]

#: The shared key variable every OpenAI-compatible client reads.
API_KEY_ENV = "OPENAI_API_KEY"

#: The shared address variable, consulted after a client's own variable.
BASE_URL_ENV = "OPENAI_BASE_URL"

#: What the three clients sent before this module existed. Kept as the fallback
#: so a box without ``OPENAI_API_KEY`` behaves byte-for-byte as it did.
PLACEHOLDER_API_KEY = "not-needed"

#: The estate's llama-swap address — the last resort for the two clients that
#: had it written into the code.
DEFAULT_BASE_URL = "http://localhost:9000/v1"


def resolve_api_key(placeholder: str = PLACEHOLDER_API_KEY) -> str:
    """The key to send: ``OPENAI_API_KEY`` when set and not blank, else the
    caller's placeholder.

    Never log, print, or interpolate the result into a message: the whole point
    of returning it here is that it goes straight into the request and nowhere
    else.
    """
    value = os.environ.get(API_KEY_ENV)
    if value is not None and value.strip():
        return value.strip()
    return placeholder


def resolve_base_url(
    *,
    explicit: Optional[str] = None,
    env_vars: Sequence[str] = (),
    default: Optional[str] = DEFAULT_BASE_URL,
    empty_env_disables: bool = False,
) -> str:
    """The address to call, by the precedence this module's docstring names.

    ``explicit`` is the per-client setting (a config value, a caller's
    argument); a blank or missing one is ignored. ``env_vars`` are the
    environment variable names to try in order — a client's own name first,
    then ``OPENAI_BASE_URL``. ``default`` is the built-in last resort; pass
    ``None`` for a client that must treat "nothing configured" as "do not call
    the model at all", and the answer is then the empty string.

    ``empty_env_disables`` is for the stamp fallback alone, whose oldest rule is
    that the FIRST of its variables that is *present* decides even when its
    value is empty — an empty value there means "deliberately switched off", and
    it never falls through to the next name. Everywhere else a blank value is
    simply skipped.
    """
    if explicit is not None and explicit.strip():
        return explicit.strip()
    for name in env_vars:
        raw = os.environ.get(name)
        if raw is None:
            continue
        if raw.strip():
            return raw.strip()
        if empty_env_disables:
            return ""
    return default or ""
