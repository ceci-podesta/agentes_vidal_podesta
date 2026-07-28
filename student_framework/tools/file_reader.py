from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

_ALLOWED_DIR = Path("sample_files").resolve()


def _list_files(directory: Path) -> str:
    """Devuelve los nombres de archivos en `directory` como string legible."""
    files = sorted(p.name for p in directory.iterdir() if p.is_file())
    return ", ".join(repr(f) for f in files) if files else "(ninguno)"


def file_reader(
    path: Annotated[str, Field(description="Ruta relativa dentro de sample_files al archivo de texto a leer.")],
) -> str:
    """Lee un archivo de texto UTF-8 dentro de sample_files y devuelve su contenido."""
    # 1. Ruta vacía o en blanco.
    if not path or not path.strip():
        available = _list_files(_ALLOWED_DIR)
        return (
            "Error: la ruta está vacía. "
            "Proporcione una ruta relativa dentro de 'sample_files', "
            f"por ejemplo: 'notas.txt'. Archivos disponibles: {available}."
        )

    # 2. Ruta absoluta.
    if Path(path).is_absolute():
        return (
            f"Error: la ruta '{path}' es absoluta. "
            "Solo se aceptan rutas relativas dentro de 'sample_files', "
            "por ejemplo: 'notas.txt' o 'subdir/archivo.md'."
        )

    # 3. Traversal con '..': podría escapar el sandbox.
    if ".." in Path(path).parts:
        return (
            f"Error: la ruta '{path}' contiene '..', lo que podría escapar del "
            "directorio permitido. Use rutas relativas sin '..'."
        )

    try:
        requested_path = (_ALLOWED_DIR / path).resolve()

        # 4. Escape del sandbox (captura symlinks y otros casos tras resolve).
        if not requested_path.is_relative_to(_ALLOWED_DIR):
            return (
                f"Error: la ruta '{path}' escapa del directorio permitido 'sample_files'. "
                "Solo se pueden leer archivos dentro de 'sample_files'."
            )

        # 5. La ruta apunta a un directorio.
        if requested_path.is_dir():
            contents = sorted(p.name for p in requested_path.iterdir())
            contents_str = ", ".join(repr(c) for c in contents) if contents else "(vacío)"
            return (
                f"Error: '{path}' es un directorio, no un archivo. "
                f"Contenido del directorio: {contents_str}."
            )

        return requested_path.read_text(encoding="utf-8")

    except FileNotFoundError:
        # Listar archivos del directorio contenedor si es válido y existe.
        parent_abs = (_ALLOWED_DIR / Path(path).parent).resolve()
        if parent_abs.is_relative_to(_ALLOWED_DIR) and parent_abs.is_dir():
            available = _list_files(parent_abs)
            parent_label = str(Path(path).parent) if str(Path(path).parent) != "." else "sample_files"
            return (
                f"Error: el archivo '{path}' no existe. "
                f"Archivos disponibles en '{parent_label}': {available}."
            )
        return f"Error: el archivo '{path}' no existe y el directorio contenedor tampoco."

    except UnicodeDecodeError:
        return f"Error: '{path}' no parece ser un archivo de texto UTF-8."
    except Exception as e:
        return f"Error al leer el archivo: {e}"


file_reader_schema = ToolSchema.from_callable(file_reader)
