"""The two QA seat clients read the key and the address from the shared rule.

``guardkit/qa/qav_shadow.py`` and ``guardkit/qa/review_seat.py`` both build an
OpenAI-compatible client. Both used to send the literal placeholder key, and the
review seat had ``http://localhost:9000/v1`` written into the code. These tests
pin what each one now sends:

* the key is ``OPENAI_API_KEY`` when that variable is set, and the old
  placeholder when it is not (a machine without the variable behaves exactly as
  it did before);
* the address follows the precedence the modules' docstrings name.

No key value is ever logged or printed here — the dummies are obvious
non-secrets, compared and discarded.
"""

import sys
import types

import pytest

import guardkit.qa.qav_shadow as qs
import guardkit.qa.review_seat as rs
from guardkit.lib.client_env import API_KEY_ENV, BASE_URL_ENV

DUMMY_KEY = "dummy-key-for-tests"
PLACEHOLDER = "not-needed"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts from a machine with none of these variables set."""
    for name in (
        API_KEY_ENV,
        BASE_URL_ENV,
        qs.QAV_SHADOW_URL_ENV,
        rs.REVIEW_SEAT_URL_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def captured_client(monkeypatch):
    """A fake ``openai.OpenAI`` that records what it was handed and stops there."""
    seen = {}

    class _FakeClient:
        def __init__(self, base_url=None, api_key=None, timeout=None):
            seen["base_url"] = base_url
            seen["api_key"] = api_key
            raise RuntimeError("stop before any network")

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = _FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    return seen


def _drive_qav(endpoint):
    call = qs._default_seat_call(endpoint)
    with pytest.raises(RuntimeError):
        call("s", "u", "m", 1.0)


def _drive_review(base_url=None):
    call = rs._default_seat_call(base_url)
    with pytest.raises(RuntimeError):
        call("s", "u", "m")


# ===========================================================================
# The QAV shadow
# ===========================================================================


def test_qav_sends_the_key_from_the_environment(captured_client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, DUMMY_KEY)
    _drive_qav(qs.DEFAULT_ENDPOINT)
    assert captured_client["api_key"] == DUMMY_KEY


def test_qav_falls_back_to_the_placeholder_key(captured_client):
    _drive_qav(qs.DEFAULT_ENDPOINT)
    assert captured_client["api_key"] == PLACEHOLDER


def test_qav_address_default_when_nothing_is_configured():
    assert qs._endpoint({}) == qs.DEFAULT_ENDPOINT
    assert qs.DEFAULT_ENDPOINT == "http://localhost:9000/v1"


def test_qav_address_openai_base_url_beats_the_default(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    assert qs._endpoint({}) == "http://shared:4000/v1"


def test_qav_address_own_variable_beats_openai_base_url(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    monkeypatch.setenv(qs.QAV_SHADOW_URL_ENV, "http://mine:4100/v1")
    assert qs._endpoint({}) == "http://mine:4100/v1"


def test_qav_address_config_block_beats_every_variable(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    monkeypatch.setenv(qs.QAV_SHADOW_URL_ENV, "http://mine:4100/v1")
    assert qs._endpoint({"endpoint": "http://config:4200/v1"}) == "http://config:4200/v1"


def test_qav_clean_env_behaves_exactly_as_before(captured_client):
    """Byte-for-byte regression: with none of these variables set the shadow
    sends the placeholder key to llama-swap, as it always has."""
    endpoint = qs._endpoint({})
    _drive_qav(endpoint)
    assert endpoint == "http://localhost:9000/v1"
    assert captured_client["base_url"] == "http://localhost:9000/v1"
    assert captured_client["api_key"] == "not-needed"


# ===========================================================================
# The code-review seat
# ===========================================================================


def test_review_seat_sends_the_key_from_the_environment(captured_client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, DUMMY_KEY)
    _drive_review()
    assert captured_client["api_key"] == DUMMY_KEY


def test_review_seat_falls_back_to_the_placeholder_key(captured_client):
    _drive_review()
    assert captured_client["api_key"] == PLACEHOLDER


def test_review_seat_address_default_when_nothing_is_configured():
    assert rs.resolve_seat_base_url() == rs.DEFAULT_BASE_URL
    assert rs.DEFAULT_BASE_URL == "http://localhost:9000/v1"


def test_review_seat_address_openai_base_url_beats_the_default(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    assert rs.resolve_seat_base_url() == "http://shared:4000/v1"


def test_review_seat_address_own_variable_beats_openai_base_url(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    monkeypatch.setenv(rs.REVIEW_SEAT_URL_ENV, "http://mine:4100/v1")
    assert rs.resolve_seat_base_url() == "http://mine:4100/v1"


def test_review_seat_address_an_explicit_value_beats_every_variable(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    monkeypatch.setenv(rs.REVIEW_SEAT_URL_ENV, "http://mine:4100/v1")
    assert rs.resolve_seat_base_url("http://caller:4200/v1") == "http://caller:4200/v1"


def test_review_seat_call_follows_the_resolved_address(captured_client, monkeypatch):
    monkeypatch.setenv(rs.REVIEW_SEAT_URL_ENV, "http://mine:4100/v1")
    _drive_review()
    assert captured_client["base_url"] == "http://mine:4100/v1"


def test_review_seat_clean_env_behaves_exactly_as_before(captured_client):
    """Byte-for-byte regression: with none of these variables set the seat call
    goes to llama-swap with the placeholder key, as it always has."""
    _drive_review()
    assert captured_client["base_url"] == "http://localhost:9000/v1"
    assert captured_client["api_key"] == "not-needed"
