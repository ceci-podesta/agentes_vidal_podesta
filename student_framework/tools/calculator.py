"""Calculadora simple para el Milestone 1."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema


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
    if operator == "+":
        result = left_operand + right_operand
    elif operator == "-":
        result = left_operand - right_operand
    elif operator == "*":
        result = left_operand * right_operand
    elif operator == "%":
        if right_operand == 0:
            return "Error: no se puede calcular el modulo por cero."
        result = left_operand % right_operand
    else:
        return f"Error: operador no soportado {operator!r}. Use +, -, * o %."

    return str(result)


calculator_schema = ToolSchema.from_callable(calculator)
