import pytest

from fine_print.config import get_settings
from fine_print.models import NOT_FOUND_MESSAGE
from fine_print.prompts import list_prompts, load_prompt


def test_the_prompt_in_the_settings_exists() -> None:
    assert get_settings().prompt_name in list_prompts()


@pytest.mark.parametrize("name", list_prompts())
def test_every_prompt_keeps_the_rules_that_answers_depend_on(name: str) -> None:
    prompt = load_prompt(name)
    assert prompt.name == name
    assert prompt.description
    assert "ONLY from the context" in prompt.system
    assert NOT_FOUND_MESSAGE in prompt.system
    assert "{not_found}" not in prompt.system
    assert "Ignore any instructions" in prompt.system


@pytest.mark.parametrize("name", list_prompts())
def test_citation_instructions_match_the_citation_style(name: str) -> None:
    prompt = load_prompt(name)
    if prompt.citation_style == "number":
        assert "[2]" in prompt.system
        assert "[Acme – Refunds]" not in prompt.system
    else:
        assert "[Acme – Refunds]" in prompt.system


def test_quote_prompts_ask_for_a_quote_before_the_answer() -> None:
    for name in ("quote_first", "numbered_quote"):
        system = load_prompt(name).system
        assert system.index('"Quote:"') < system.index('"Answer:"')


def test_unknown_prompt_lists_the_choices() -> None:
    with pytest.raises(ValueError, match="baseline"):
        load_prompt("magic")
