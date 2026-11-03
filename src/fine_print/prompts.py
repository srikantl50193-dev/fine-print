"""Load the prompt variants, which are version-controlled files in the prompts folder."""

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fine_print.models import NOT_FOUND_MESSAGE, FrozenModel

PROMPTS_DIR = Path(__file__).parent / "prompts"

type CitationStyle = Literal["label", "number"]


class Prompt(FrozenModel):
    """A system prompt, and how the replies it produces cite their passages."""

    name: str
    description: str
    citation_style: CitationStyle
    system: str


def list_prompts() -> list[str]:
    """Return the names of the available prompts, sorted."""
    return sorted(path.stem for path in PROMPTS_DIR.glob("*.toml"))


@lru_cache
def load_prompt(name: str) -> Prompt:
    """Read one prompt, filling in the exact wording of the refusal.

    Raises:
        ValueError: If there is no prompt with this name.
    """
    if name not in list_prompts():
        raise ValueError(f"Unknown prompt '{name}'. Choose from: {', '.join(list_prompts())}")
    data = tomllib.loads((PROMPTS_DIR / f"{name}.toml").read_text(encoding="utf-8"))
    data["system"] = data["system"].replace("{not_found}", NOT_FOUND_MESSAGE)
    return Prompt(name=name, **data)
