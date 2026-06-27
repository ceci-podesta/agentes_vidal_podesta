from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


def file_reader(
    path: Annotated[str, Field(description="Ruta al archivo de texto a leer.")],
) -> str:
    """Lee el contenido completo de un archivo de texto en UTF-8 y lo devuelve como string."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: no se encontró el archivo '{path}'."
    except IsADirectoryError:
        return f"Error: '{path}' es un directorio, no un archivo."
    except Exception as e:
        return f"Error al leer el archivo: {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
