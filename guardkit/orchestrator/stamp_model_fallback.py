"""THE MODEL FALLBACK — the one place a model may decide a routing-law stamp.

What this is, in one paragraph. Every scenario in a feature must say how it
will be proved, with one word from a closed list (``toolchain``, ``hurl``,
``exam``, ``probe:bus``, ``probe:process``, ``flutter``, ``playwright``,
``operator``). Rules R1–R10 in ``stamp_normalizer`` decide almost all of them
from the scenario's own text. When no rule matches, the law refuses loudly and
names the title rather than inventing a stamp. This module is what happens
next, and only then: the refused titles — and nothing else — are handed to a
model with the closed list and the rules' own summary, and the model answers
one word per title. The answer is checked against the closed list. Anything
that is not a clean answer leaves the titles exactly as they were: refused.

Design of record: ``ai-transition/docs/routing-law-stamp-normalizer-rules-
2026-08-15.md``, the paragraph "What the model is asked, when asked" ("only
the refused titles, with the vocabulary, the R1–R10 table as its rationale,
and the instruction to answer ONE word per title — then the answer is
validated against the closed list (bogus → the same loud rejection the law
already emits). If the model's answer is ``operator``, the card says so and
Rich sees it — an operator stamp is never silent."). Ruled by Rich
2026-08-31 (repair item 11): hand-widening the rules cannot keep up — clause
(h) was widened on 2026-08-28 for concurrency phrasings and still refused
"Concurrent requests return the same 7-day data" ("the same", not
"identical") and "Concurrent deactivation requests are handled idempotently"
("idempotently", not "gracefully"). The spec seat's vocabulary outruns a
hand-maintained synonym list.

THE PROPERTY THAT MATTERS MOST — failure is the old behaviour, never a guess.
No model configured, an endpoint that cannot be reached, a timeout, an HTTP
error, a malformed reply, an answer with the wrong number of lines, a word
outside the closed list: every one of these leaves the titles refused and
writes one plain line saying the model could not be asked (or that its answer
was rejected). A stamp is never invented. A silent default would let unproved
scenarios through the routing law, which is the exact thing the law exists to
prevent.

Configuration (environment variables)
-------------------------------------
``GUARDKIT_STAMP_MODEL_URL`` — the OpenAI-compatible endpoint, the ``/v1``
root. When it is not set, ``OPENAI_BASE_URL`` is used instead; on this estate
that is llama-swap at ``http://localhost:9000/v1``. There is deliberately NO
built-in default: with neither set (or either set to an empty value) the model
is NOT configured, it is never called, and the behaviour is exactly today's —
refuse loud.

``GUARDKIT_STAMP_MODEL`` — the model name to ask. Default
``qwen36-workhorse`` (the estate's general workhorse seat).

``GUARDKIT_STAMP_MODEL_TIMEOUT_S`` — seconds to wait for the answer. Default
15, clamped to 1–60 so a hung endpoint can never stall a planning run. An
unreadable value falls back to the default with a warning.

``OPENAI_API_KEY`` — the key sent with the call. When it is set and not blank
that value is sent; otherwise the placeholder ``not-needed`` the estate's
llama-swap has always ignored, so a box without the variable behaves exactly as
before. The key is never logged or printed. Address and key are both resolved
by the one shared rule in ``guardkit/lib/client_env.py``, whose precedence for
this client is: ``GUARDKIT_STAMP_MODEL_URL``, then ``OPENAI_BASE_URL``, then
nothing (not configured — the model is never asked).

The call is INJECTED. ``decide_refused_titles(..., ask_model=…)`` takes a
callable ``(prompt) -> answer text``; the default one is built from the
environment above. Every test drives a fake and nothing reaches the network.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from guardkit.lib.client_env import resolve_api_key, resolve_base_url
from guardkit.orchestrator.verifier_stamp import VERIFIER_HOMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: The endpoint, preferred name (the ``/v1`` root of an OpenAI-compatible API).
MODEL_URL_ENV = "GUARDKIT_STAMP_MODEL_URL"

#: The endpoint, fallback name — the estate's usual one (llama-swap).
MODEL_URL_FALLBACK_ENV = "OPENAI_BASE_URL"

#: What the estate serves on, quoted in the "not configured" line so the
#: reader knows what to set. NOT a default: nothing is assumed.
EXAMPLE_ENDPOINT = "http://localhost:9000/v1"

#: The model name to ask.
MODEL_NAME_ENV = "GUARDKIT_STAMP_MODEL"
DEFAULT_MODEL_NAME = "qwen36-workhorse"

#: How long to wait for the answer (seconds).
MODEL_TIMEOUT_ENV = "GUARDKIT_STAMP_MODEL_TIMEOUT_S"
MODEL_MAX_TOKENS_ENV = "GUARDKIT_STAMP_MODEL_MAX_TOKENS"
DEFAULT_TIMEOUT_S = 180.0
#: Long, deliberately. The estate serves one model at a time and swaps them in on
#: demand, so the first call after a swap waits for a 35-billion-parameter model to
#: load — about a minute and a half on this box. A 15 second timeout (the first
#: value here) never once reached a cold model: every call timed out, every title
#: stayed refused, and the whole mechanism would have looked safe while doing
#: nothing. Measured 2026-08-31. One stamp decision per feature, inside a planning
#: run that already takes fifteen minutes, is worth the wait.
MIN_TIMEOUT_S = 1.0
MAX_TIMEOUT_S = 60.0

#: One word per title, so the reply is tiny. Kept small on purpose: a model
#: that starts writing prose runs out of room and its answer is rejected.
MAX_ANSWER_TOKENS = 8192
#: Raised from 2048 on 2026-08-31 after a live refusal: the deletion titles in
#: sentence 6 of the exam needed more thinking than the concurrency titles, and at
#: 2048 the reply came back empty again. 8192 answered them. The budget is for the
#: model's reasoning, not the answer — the answer is one word.
#: Generous for an answer of a few words, because the seat that answers is a
#: reasoning model: it writes its thinking first and the answer last, both out of
#: the same budget. At the first value here the thinking used the whole budget and
#: the answer came back EMPTY every time — a live call that looked like a model
#: refusing to answer, when it had simply been cut off mid-thought. Measured
#: 2026-08-31 against qwen36-workhorse.

#: What a model-decided stamp is recorded as, wherever a rule id is recorded
#: (``NormalizeResult.rules``). Never an R-number: nobody may mistake a
#: model-decided stamp for a rule-decided one.
MODEL_RULE = "model"

#: The comment written above a model-decided stamp in the feature YAML.
MODEL_STAMP_COMMENT = "# stamped by the model — no rule (R1-R10) could decide this title"

#: A prompt -> the model's raw answer text.
ModelAsker = Callable[[str], str]


class ModelAnswerRejected(ValueError):
    """The model answered, but the answer is not usable — the wrong number of
    lines, an empty line, or a word outside the closed list. The titles stay
    refused; nothing is stamped."""


# ---------------------------------------------------------------------------
# The rules' own summary, read from the rules module so it cannot drift
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleSummary:
    """One rule as the model is told about it."""

    rule: str
    home: str
    description: str


# The rule table in ``stamp_normalizer``'s module docstring: lines that start
# at column 0 with an R-number, each ending in ``→ ``home``` (possibly some
# lines later), and the block closing with the "no rule matched" line.
_RULE_START_RE = re.compile(r"^(R\d{1,2})\s+(\S.*)$")
_TABLE_END_RE = re.compile(r"^—")
_HOME_RE = re.compile(r"→\s*``([a-z:]+)``")
_ARROW_SPAN_RE = re.compile(r"→\s*``[a-z:]+``")


def rule_table(doc: Optional[str] = None) -> List[RuleSummary]:
    """The R1–R10 summary, READ FROM the rules module's own docstring.

    Derived, not copied: the summary the model is given is the same text that
    sits above the regex families in ``stamp_normalizer``, so the two cannot
    drift apart. If the table cannot be read (fewer than the ten rules, or a
    home outside the closed list) this raises — and the caller treats that as
    "the model could not be asked", which is the old behaviour: refuse loud.
    """
    if doc is None:
        from guardkit.orchestrator import stamp_normalizer  # local: avoids a cycle

        doc = stamp_normalizer.__doc__ or ""

    chunks: List[List[str]] = []
    ids: List[str] = []
    collecting = False
    for raw in doc.splitlines():
        start = _RULE_START_RE.match(raw)
        if start:
            collecting = True
            ids.append(start.group(1))
            chunks.append([start.group(2)])
            continue
        if not collecting:
            continue
        if _TABLE_END_RE.match(raw) or (raw and not raw[0].isspace()):
            break  # the table ends at the "no rule matched" row
        if raw.strip():
            chunks[-1].append(raw.strip())

    summaries: List[RuleSummary] = []
    for rule, lines in zip(ids, chunks):
        text = " ".join(lines)
        home_match = _HOME_RE.search(text)
        if not home_match:
            raise ValueError(f"the rule table row for {rule} names no home")
        home = home_match.group(1)
        if home not in VERIFIER_HOMES:
            raise ValueError(f"the rule table row for {rule} names {home!r}, which is not one of the allowed words")
        description = _ARROW_SPAN_RE.sub("", text).replace("``", "").replace("**", "")
        description = re.sub(r"\s+", " ", description).strip(" +")
        summaries.append(RuleSummary(rule=rule, home=home, description=description))

    if len(summaries) != 10:
        raise ValueError(
            f"the rule table should carry ten rules (R1-R10), read {len(summaries)}"
        )
    return summaries


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


#: The closed list minus the one word the model cannot deliver. A ``toolchain``
#: stamp must name the test node that proves the scenario; the model has no node
#: to name, so offering it invites an answer that cannot be written.
OFFERABLE_HOMES = tuple(h for h in VERIFIER_HOMES if h != "toolchain")


def build_prompt(titles: Sequence[str], *, rules: Optional[Sequence[RuleSummary]] = None) -> str:
    """The exact text sent to the model: the closed list, the rules' own
    summary as the rationale for what each word means, the refused titles, and
    the instruction to answer one word per title."""
    table = list(rules) if rules is not None else rule_table()
    count = len(titles)
    lines: List[str] = [
        "A build system proves every test scenario in exactly ONE way.",
        "",
        "The closed list of ways — your answer must use these words and nothing else:",
        "  " + ", ".join(OFFERABLE_HOMES),
        "",
        "The rules below mention one more way, toolchain, which is NOT available"
        " to you: it has to name the particular test that proves the scenario, and"
        " you have no test to name. Never answer toolchain.",
        "",
        "What each way means. These are the rules the build system applies first,"
        " in the order it applies them. None of them matched the titles below,"
        " which is why you are being asked:",
    ]
    for entry in table:
        lines.append(f"  {entry.rule} -> {entry.home}: {entry.description}")
    lines += [
        "",
        f"Decide the way to prove each of these {count} scenario title(s):",
    ]
    for number, title in enumerate(titles, 1):
        lines.append(f"  {number}. {title}")
    lines += [
        "",
        f"Answer with exactly {count} line(s), one line per title, in the same order.",
        "Each line is ONE word from the closed list and nothing else: no numbering,"
        " no punctuation, no explanation, no blank lines.",
        "Answer operator only for work a person has to do by hand.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

_THINK_CLOSE = "</think>"


def parse_answer(text: object, titles: Sequence[str]) -> Dict[str, str]:
    """``{title: word}`` when the answer is exactly one allowed word per title,
    in order. Anything else raises :class:`ModelAnswerRejected` — and the
    caller keeps the loud refusal the law already emits.

    The one allowance: a model that thinks out loud wraps its reasoning in
    ``<think>…</think>``; everything up to the last closing tag is the
    harness's wrapper, not the answer, and is dropped before checking. What
    is left must still be exactly one allowed word per title.
    """
    if not isinstance(text, str):
        raise ModelAnswerRejected(
            f"the model's answer was not text (it was {type(text).__name__})"
        )
    answer = text
    if _THINK_CLOSE in answer:
        answer = answer.rsplit(_THINK_CLOSE, 1)[1]
    words = [line.strip() for line in answer.strip().splitlines()]
    words = [word for word in words if word]
    if len(words) != len(titles):
        raise ModelAnswerRejected(
            f"expected {len(titles)} answer(s), one word per title, but the "
            f"model gave {len(words)}: {words!r}"
        )
    decided: Dict[str, str] = {}
    for title, word in zip(titles, words):
        home = word.lower()
        if home not in VERIFIER_HOMES:
            raise ModelAnswerRejected(
                f"the model answered {word!r} for {title!r}, which is not one of "
                f"the allowed words ({', '.join(OFFERABLE_HOMES)})"
            )
        if home not in OFFERABLE_HOMES:
            # A toolchain stamp must name the test node that proves the scenario,
            # and the model has no node to name — so it is not offered above, and
            # is refused here too if it is answered anyway. Without this the
            # answer is accepted, the writer then rejects it for the missing node,
            # and the run dies mid-write with a validation error instead of the
            # plain refusal the law is supposed to give (found by the coach, 08-31).
            raise ModelAnswerRejected(
                f"the model answered {word!r} for {title!r}; that way of proving a "
                f"scenario has to name the test that proves it, and the model has "
                f"no test to name, so the title stays refused"
            )
        decided[title] = home
    return decided


# ---------------------------------------------------------------------------
# The default call (the impure edge — injected everywhere else)
# ---------------------------------------------------------------------------


def _endpoint() -> str:
    """The configured endpoint, or "" when none is configured. An empty value
    means NOT configured — it never falls through to the other name.

    Resolved through the one shared rule
    (:func:`guardkit.lib.client_env.resolve_base_url`): ``GUARDKIT_STAMP_MODEL_URL``
    first, then ``OPENAI_BASE_URL``, and deliberately no built-in default."""
    return resolve_base_url(
        env_vars=(MODEL_URL_ENV, MODEL_URL_FALLBACK_ENV),
        default=None,
        empty_env_disables=True,
    )


def _timeout_seconds() -> float:
    raw = os.environ.get(MODEL_TIMEOUT_ENV, "").strip()
    if not raw:
        return DEFAULT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "STAMP NORMALIZER: %s=%r is not a number of seconds — using %ss",
            MODEL_TIMEOUT_ENV,
            raw,
            DEFAULT_TIMEOUT_S,
        )
        return DEFAULT_TIMEOUT_S
    return max(MIN_TIMEOUT_S, min(MAX_TIMEOUT_S, value))


def completions_url(base_url: str) -> str:
    """``http://host:9000/v1`` -> ``http://host:9000/v1/chat/completions``
    (an endpoint already spelled out to the completions path is left alone)."""
    root = base_url.strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return root + "/chat/completions"


def build_default_asker(model_name: Optional[str] = None) -> Optional[ModelAsker]:
    """The environment's model call, or ``None`` when no endpoint is
    configured (in which case the model is never asked and the titles stay
    refused)."""
    base = _endpoint()
    if not base:
        return None
    model = (model_name or os.environ.get(MODEL_NAME_ENV, "") or DEFAULT_MODEL_NAME).strip()
    url = completions_url(base)
    timeout = _timeout_seconds()

    def _ask(prompt: str) -> str:
        body = json.dumps(
            {
                "model": model,
                "temperature": 0.0,
                "max_tokens": int(os.environ.get(MODEL_MAX_TOKENS_ENV, "") or MAX_ANSWER_TOKENS),
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                # The key the estate's router expects: OPENAI_API_KEY when it is
                # set, else the placeholder llama-swap has always ignored. Never
                # logged — it goes into the request and nowhere else.
                "Authorization": f"Bearer {resolve_api_key()}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"the reply from {url} carried no answer: {payload!r}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError(f"the reply from {url} carried no answer text: {payload!r}")
        if not content.strip():
            # A reasoning model that ran out of budget mid-thought answers with an
            # empty string. Say so, rather than letting it look like a refusal to
            # answer: the two need different fixes.
            reason = message.get("reasoning_content") if isinstance(message, dict) else None
            raise ValueError(
                f"the reply from {url} was empty"
                + (f" — the model was still thinking when it ran out of room "
                   f"(raise {MODEL_MAX_TOKENS_ENV} above {MAX_ANSWER_TOKENS})"
                   if isinstance(reason, str) and reason.strip() else "")
            )
        return content

    return _ask


# ---------------------------------------------------------------------------
# The one entry point the normalizer calls
# ---------------------------------------------------------------------------


def decide_refused_titles(
    titles: Sequence[str],
    *,
    ask_model: Optional[ModelAsker] = None,
    feature_id: str = "",
) -> Dict[str, str]:
    """Ask the model about titles NO RULE COULD DECIDE, and return
    ``{title: word}`` for every one it decided.

    Never raises. An empty result means "keep the refusal exactly as it was" —
    which is what happens for every failure: no endpoint configured, an
    endpoint that cannot be reached, a timeout, an HTTP error, a malformed
    reply, or an answer that is not one allowed word per title. Each of those
    writes ONE plain line saying the model could not be asked, or that its
    answer was rejected. A stamp is never invented.

    All or nothing: a single bad word rejects the whole answer, so a model can
    only ever turn a refusal into a word from the closed list.
    """
    wanted = list(dict.fromkeys(titles))
    if not wanted:
        return {}
    where = f"feature {feature_id}: " if feature_id else ""

    asker = ask_model or build_default_asker()
    if asker is None:
        logger.warning(
            "STAMP NORMALIZER: %sthe model could not be asked about %d title(s) "
            "no rule could decide — no model endpoint is configured (set %s, or "
            "%s, to something like %s). The titles stay refused and nothing was "
            "stamped.",
            where,
            len(wanted),
            MODEL_URL_FALLBACK_ENV,
            MODEL_URL_ENV,
            EXAMPLE_ENDPOINT,
        )
        return {}

    try:
        prompt = build_prompt(wanted)
        raw = asker(prompt)
    except Exception as exc:  # noqa: BLE001 — every failure is the old behaviour
        logger.warning(
            "STAMP NORMALIZER: %sthe model could not be asked about %d title(s) "
            "no rule could decide (%s: %s). The titles stay refused and nothing "
            "was stamped.",
            where,
            len(wanted),
            type(exc).__name__,
            exc,
        )
        return {}

    try:
        decided = parse_answer(raw, wanted)
    except ModelAnswerRejected as exc:
        logger.warning(
            "STAMP NORMALIZER: %sthe model's answer was rejected — %s. The %d "
            "title(s) stay refused and nothing was stamped.",
            where,
            exc,
            len(wanted),
        )
        return {}

    logger.info(
        "STAMP NORMALIZER: %sthe model decided %d title(s) no rule could decide: %s",
        where,
        len(decided),
        "; ".join(f"{title!r} -> {home}" for title, home in decided.items()),
    )
    return decided


__all__ = [
    "MODEL_URL_ENV",
    "MODEL_URL_FALLBACK_ENV",
    "MODEL_NAME_ENV",
    "DEFAULT_MODEL_NAME",
    "MODEL_TIMEOUT_ENV",
    "DEFAULT_TIMEOUT_S",
    "EXAMPLE_ENDPOINT",
    "MAX_ANSWER_TOKENS",
    "MODEL_RULE",
    "MODEL_STAMP_COMMENT",
    "ModelAsker",
    "ModelAnswerRejected",
    "RuleSummary",
    "rule_table",
    "build_prompt",
    "parse_answer",
    "completions_url",
    "build_default_asker",
    "decide_refused_titles",
]
