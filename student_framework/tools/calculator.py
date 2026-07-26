"""Calculadora simple para el Milestone 1."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

_ALLOWED_OPERATORS = ("+", "-", "*", "%")


def calculator(
    left_operand: Annotated[
        float,
        Field(description="Primer operando numerico."),
    ],
    right_operand: Annotated[
        float,
        Field(description="Segundo operando numerico."),
    ],
    operator: Annotated[
        str,
        Field(description="Operador aritmetico soportado: +, -, * o % (modulo)."),
    ],
) -> str:
    """Calcula el resultado de una operacion aritmetica binaria simple."""
    # Validar operandos en tiempo de ejecucion: el LLM puede enviar strings aunque
    # el schema declare float, y Python no coerciona tipos en tiempo de ejecucion.
    try:
        left_val = float(left_operand)
    except (TypeError, ValueError):
        return (
            f"Error: el parámetro 'left_operand' recibió el valor {left_operand!r}, "
            "que no es numérico. Se esperaba un número entero o decimal, "
            "por ejemplo: 3, 2.5 o -10."
        )

    try:
        right_val = float(right_operand)
    except (TypeError, ValueError):
        return (
            f"Error: el parámetro 'right_operand' recibió el valor {right_operand!r}, "
            "que no es numérico. Se esperaba un número entero o decimal, "
            "por ejemplo: 3, 2.5 o -10."
        )

    if operator not in _ALLOWED_OPERATORS:
        allowed = ", ".join(f"'{op}'" for op in _ALLOWED_OPERATORS)
        return (
            f"Error: el operador {operator!r} no está soportado. "
            f"Los operadores permitidos son: {allowed}."
        )

    if operator == "+":
        result = left_val + right_val
    elif operator == "-":
        result = left_val - right_val
    elif operator == "*":
        result = left_val * right_val
    else:  # operator == "%"
        if right_val == 0:
            return (
                "Error: no se puede calcular el módulo cuando 'right_operand' es cero. "
                "El divisor debe ser un valor distinto de cero."
            )
        result = left_val % right_val

    # Evitar el sufijo ".0" innecesario cuando el resultado es un entero.
    if result == int(result):
        return str(int(result))
    return str(result)


calculator_schema = ToolSchema.from_callable(calculator)
