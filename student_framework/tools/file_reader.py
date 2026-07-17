from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

_ALLOWED_DIR = Path("sample_files").resolve()


def file_reader(
    path: Annotated[str, Field(description="Ruta relativa dentro de sample_files al archivo de texto a leer.")],
) -> str:
    """Lee un archivo de texto UTF-8 dentro de sample_files y devuelve su contenido."""
    try:
        requested_path = (_ALLOWED_DIR / path).resolve()

        if not requested_path.is_relative_to(_ALLOWED_DIR):
            return "Error: solo se pueden leer archivos dentro de sample_files."

        if requested_path.is_dir():
            return f"Error: '{path}' es un directorio, no un archivo."

        return requested_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: no se encontró el archivo '{path}'."
    except UnicodeDecodeError:
        return f"Error: '{path}' no parece ser un archivo de texto UTF-8."
    except Exception as e:
        return f"Error al leer el archivo: {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
