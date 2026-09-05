"""The one rule for the seat key and the seat address (``guardkit/lib/client_env``).

Three clients used to send the literal placeholder key and two had the address
written into the code. These tests pin the rule they now share: the key comes
from ``OPENAI_API_KEY`` when it is set and falls back to the placeholder when it
is not, and the address follows a fixed order of precedence. Nothing here ever
prints a key value: the dummies are obvious non-secrets and they are compared,
never logged.
"""

import pytest

from guardkit.lib.client_env import (
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_BASE_URL,
    PLACEHOLDER_API_KEY,
    resolve_api_key,
    resolve_base_url,
)

CLIENT_ENV = "GUARDKIT_TEST_CLIENT_URL"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts from a machine with none of these variables set."""
    for name in (API_KEY_ENV, BASE_URL_ENV, CLIENT_ENV):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# The key
# ---------------------------------------------------------------------------


def test_key_comes_from_the_environment_when_it_is_set(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "dummy-key-for-tests")
    assert resolve_api_key() == "dummy-key-for-tests"


def test_key_falls_back_to_the_placeholder_when_the_variable_is_unset():
    assert resolve_api_key() == PLACEHOLDER_API_KEY
    assert PLACEHOLDER_API_KEY == "not-needed"


def test_a_blank_key_is_the_same_as_no_key(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "   ")
    assert resolve_api_key() == PLACEHOLDER_API_KEY


def test_a_key_is_stripped_of_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "  dummy-key-for-tests\n")
    assert resolve_api_key() == "dummy-key-for-tests"


def test_a_client_may_name_its_own_placeholder(monkeypatch):
    assert resolve_api_key("its-own-placeholder") == "its-own-placeholder"
    monkeypatch.setenv(API_KEY_ENV, "dummy-key-for-tests")
    assert resolve_api_key("its-own-placeholder") == "dummy-key-for-tests"


# ---------------------------------------------------------------------------
# The address — the precedence table, one row at a time
# ---------------------------------------------------------------------------


def _resolve():
    return resolve_base_url(env_vars=(CLIENT_ENV, BASE_URL_ENV))


def test_nothing_set_gives_the_built_in_default():
    assert _resolve() == DEFAULT_BASE_URL


def test_openai_base_url_beats_the_default(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    assert _resolve() == "http://shared:4000/v1"


def test_the_clients_own_variable_beats_openai_base_url(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    monkeypatch.setenv(CLIENT_ENV, "http://mine:4100/v1")
    assert _resolve() == "http://mine:4100/v1"


def test_an_explicit_setting_beats_every_variable(monkeypatch):
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    monkeypatch.setenv(CLIENT_ENV, "http://mine:4100/v1")
    assert (
        resolve_base_url(explicit="http://config:4200/v1", env_vars=(CLIENT_ENV, BASE_URL_ENV))
        == "http://config:4200/v1"
    )


def test_a_blank_explicit_setting_is_ignored(monkeypatch):
    monkeypatch.setenv(CLIENT_ENV, "http://mine:4100/v1")
    assert resolve_base_url(explicit="  ", env_vars=(CLIENT_ENV, BASE_URL_ENV)) == "http://mine:4100/v1"


def test_a_blank_variable_is_skipped_and_the_next_one_decides(monkeypatch):
    monkeypatch.setenv(CLIENT_ENV, "")
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    assert _resolve() == "http://shared:4000/v1"


def test_no_default_means_the_answer_can_be_nothing():
    assert resolve_base_url(env_vars=(CLIENT_ENV,), default=None) == ""


def test_a_blank_variable_switches_the_client_off_when_it_is_allowed_to(monkeypatch):
    """The stamp fallback's oldest rule: the first variable that is PRESENT
    decides, even when it is empty — an empty value means "do not call"."""
    monkeypatch.setenv(CLIENT_ENV, "")
    monkeypatch.setenv(BASE_URL_ENV, "http://shared:4000/v1")
    assert (
        resolve_base_url(
            env_vars=(CLIENT_ENV, BASE_URL_ENV), default=None, empty_env_disables=True
        )
        == ""
    )
