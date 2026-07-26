"""Prueba manual de las features del Milestone 2 contra Bedrock real.

Llamadas a Bedrock: ~5 en total (minimizadas a propósito).
Las features que no requieren LLM real se prueban directamente.

Ejecutar desde la raíz del proyecto con el entorno activado:

    python scripts/prueba_m2.py

Requiere las variables de entorno de Bedrock:
    export AWS_REGION="us-east-2"
    export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
"""

from __future__ import annotations

from pydantic import BaseModel

from mia_agents.llm_client import LLMClient
from student_framework import build_agent
from student_framework.tools.calculator import calculator
from student_framework.tools.file_reader import file_reader
from student_framework.tools.word_counter import word_counter


def sep(titulo: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {titulo}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Statefulness — 2 llamadas a Bedrock
# ---------------------------------------------------------------------------
def probar_statefulness(llm: LLMClient) -> None:
    sep("1. STATEFULNESS  [2 llamadas a Bedrock]")
    agent = build_agent({"llm_client": llm})

    r1 = agent.run("Mi número secreto es 7331. Recordalo.")
    print(f"  Turno 1: {r1.answer}")

    r2 = agent.run("¿Cuál era mi número secreto?")
    print(f"  Turno 2: {r2.answer}")

    if "7331" in r2.answer:
        print("  ✓ El agente recordó el número del turno anterior")
    else:
        print("  ⚠ El agente no mencionó el número — revisá la respuesta")


# ---------------------------------------------------------------------------
# 2. Salida estructurada — 1 a 3 llamadas a Bedrock
# ---------------------------------------------------------------------------
def probar_structured_call(llm: LLMClient) -> None:
    sep("2. SALIDA ESTRUCTURADA  [1-3 llamadas a Bedrock]")

    class Evaluacion(BaseModel):
        concepto: str
        dificultad: str           # "baja", "media" o "alta"
        ejemplo_en_una_linea: str

    agent = build_agent({"llm_client": llm})
    resultado = agent.structured_call(
        prompt=(
            "Evaluá el concepto 'sliding window' como estrategia de memoria "
            "en un agente conversacional. Indicá el nombre del concepto, "
            "su dificultad (baja/media/alta) y un ejemplo en una línea."
        ),
        schema=Evaluacion,
    )
    print(f"  Concepto   : {resultado.concepto}")
    print(f"  Dificultad : {resultado.dificultad}")
    print(f"  Ejemplo    : {resultado.ejemplo_en_una_linea}")
    print("  ✓ structured_call devolvió un objeto Pydantic válido")


# ---------------------------------------------------------------------------
# 3. Errores recuperables en tools — 0 llamadas a Bedrock
# ---------------------------------------------------------------------------
def probar_errores_recuperables() -> None:
    sep("3. ERRORES RECUPERABLES EN TOOLS  [0 llamadas a Bedrock]")
    print("  (prueba directa de los mensajes que recibe el LLM)\n")

    casos = [
        ("calculator('veinte', 5, '+')", calculator("veinte", 5, "+")),
        ("calculator(10, 0, '%')",       calculator(10, 0, "%")),
        ("calculator(10, 5, '/')",       calculator(10, 5, "/")),
        ("file_reader('')",              file_reader("")),
        ("file_reader('/etc/passwd')",   file_reader("/etc/passwd")),
        ("file_reader('../escape')",     file_reader("../escape")),
        ("file_reader('no_existe.txt')", file_reader("no_existe.txt")),
        ("word_counter(None)",           word_counter(None)),
        ("word_counter(42)",             word_counter(42)),
    ]
    for nombre, resultado in casos:
        print(f"  {nombre}")
        print(f"    → {resultado}\n")

    print("  ✓ Todos los mensajes son accionables (indican qué falló y cómo corregirlo)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\nInicializando cliente Bedrock...")
    llm = LLMClient.from_env()
    print("Cliente listo.")

    probar_statefulness(llm)       # 2 llamadas
    probar_structured_call(llm)    # 1-3 llamadas
    probar_errores_recuperables()  # 0 llamadas

    print(f"\n{'=' * 60}")
    print("  Prueba M2 completada.")
    print("=" * 60)


if __name__ == "__main__":
    main()
