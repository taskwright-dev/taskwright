"""The stamp normalizer's model fallback reads the key and the address from the
shared rule (``guardkit/lib/client_env``).

The fallback used to send the literal placeholder key. Since the factory's calls
go through a router that checks the key, it now sends ``OPENAI_API_KEY`` when
that variable is set and the old placeholder when it is not. Its address rule is
unchanged and these tests say so out loud, because this client is the one that
must have NO built-in default: with nothing configured the model is never asked
and refused titles stay refused.

No key value is ever logged or printed here — the dummy is an obvious
non-secret, read off the request header and compared.
"""

import json

import pytest

import guardkit.orchestrator.stamp_model_fallback as smf
from guardkit.lib.client_env import API_KEY_ENV, BASE_URL_ENV

DUMMY_KEY = "dummy-key-for-tests"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start from a machine with none of these variables set. (The suite's own
    autouse fixture pins ``GUARDKIT_STAMP_MODEL_URL`` to an empty value so no
    test can reach a live model; these tests set it deliberately instead.)"""
    for name in (API_KEY_ENV, BASE_URL_ENV, smf.MODEL_URL_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def captured_request(monkeypatch):
    """Catch the ``urllib`` request the asker builds; answer with one word."""
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            body = {"choices": [{"message": {"content": "toolchain"}}]}
            return json.dumps(body).encode("utf-8")

    def _fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["authorization"] = request.get_header("Authorization")
        return _Response()

    monkeypatch.setattr(smf.urllib.request, "urlopen", _fake_urlopen)
    return seen


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_the_key_comes_from_the_environment(captured_request, monkeypatch):
    monkeypatch.setenv(smf.MODEL_URL_ENV, "http://mine:4100/v1")
    monkeypatch.setenv(API_KEY_ENV, DUMMY_KEY)
    smf.build_default_asker()("a prompt")
    assert captured_request["authorization"] == f"Bearer {DUMMY_KEY}"


def test_the_key_falls_back_to_the_placeholder(captured_request, monkeypatch):
    monkeypatch.setenv(smf.MODEL_URL_ENV, "http://mine:4100/v1")
    smf.build_default_asker()("a prompt")
    assert captured_request["authorization"] == "Bearer not-needed"


# ---------------------------------------------------------------------------
# The address — unchanged, and deliberately without a built-in default
# ---------------------------------------------------------------------------


def test_nothing_configured_means_the_model_is_never_asked():
    assert smf._endpoint() == ""
    assert smf.build_default_asker() is None


def test_openai_base_url_is_used_when_the_client_variable_is_absent(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    assert smf._endpoint() == "http://shared:4000/v1"


def test_the_clients_own_variable_beats_openai_base_url(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    monkeypatch.setenv(smf.MODEL_URL_ENV, "http://mine:4100/v1")
    assert smf._endpoint() == "http://mine:4100/v1"


def test_the_client_variable_set_to_an_empty_value_switches_it_off(monkeypatch):
    """Its oldest rule, kept: an empty value means "not configured" and it never
    falls through to ``OPENAI_BASE_URL``."""
    monkeypatch.setenv(smf.MODEL_URL_ENV, "")
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    assert smf._endpoint() == ""
    assert smf.build_default_asker() is None


def test_clean_env_behaves_exactly_as_before(captured_request, monkeypatch):
    """Byte-for-byte regression: with only the endpoint set, as on the box
    today, the call goes to that endpoint with the placeholder key."""
    monkeypatch.setenv(BASE_URL_ENV, "http://localhost:9000/v1")
    smf.build_default_asker()("a prompt")
    assert captured_request["url"] == "http://localhost:9000/v1/chat/completions"
    assert captured_request["authorization"] == "Bearer not-needed"
