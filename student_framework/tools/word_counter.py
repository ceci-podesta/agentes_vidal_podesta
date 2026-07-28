from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def word_counter(
    text: Annotated[str, Field(description="El texto cuyas palabras se desean contar.")],
) -> str:
    """Cuenta la cantidad de palabras en un texto y devuelve el resultado."""
    if text is None:
        return (
            "Error: el parámetro 'text' es nulo. "
            "Se esperaba una cadena de texto, por ejemplo: 'hola mundo'."
        )
    if not isinstance(text, str):
        return (
            f"Error: el parámetro 'text' recibió un valor de tipo {type(text).__name__!r} "
            f"({text!r}), que no es texto. Se esperaba una cadena de caracteres."
        )
    words = text.split()
    return str(len(words))


word_counter_schema = ToolSchema.from_callable(word_counter)
