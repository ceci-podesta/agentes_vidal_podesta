from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def word_counter(
    text: Annotated[str, Field(description="El texto cuyas palabras se desean contar.")],
) -> str:
    """Cuenta la cantidad de palabras en un texto y devuelve el resultado."""
    words = text.split()
    return str(len(words))


word_counter_schema = ToolSchema.from_callable(word_counter)
