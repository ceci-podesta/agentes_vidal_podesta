"""Planificador inicial opcional para los escenarios M3.

Genera, con una única llamada a `structured_call` (mecanismo obligatorio de
M2: fuerza `final_result`, no admite texto libre), un plan ordenado de
pasos en lenguaje natural a partir del mensaje del usuario, antes de que el
agente haya observado el mundo. No reemplaza el loop ReAct: se inyecta como
un bloque más en el system prompt, junto al scratchpad, y el agente sigue
decidiendo tool call por tool call.

Motivación (ver ENUNCIADO_M3.md, `office-sequence`): un goal compuesto y
ordenado premia descomponer el objetivo en subpasos en vez de reaccionar
turno a turno. Además, al forzar `final_result` en este primer paso, el
agente no puede "contestar solo con texto" — el fallo más grave observado en
`color-locks` (el LLM a veces devuelve una narración completa sin llamar a
ninguna tool porque el mensaje inicial ya describe la escena en detalle).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Lista ordenada de pasos en lenguaje natural para alcanzar el objetivo."""

    steps: list[str] = Field(
        min_length=1,
        description=(
            "Pasos ordenados, en lenguaje natural, para cumplir el "
            "objetivo. No uses nombres de tools ni IDs: todavía no "
            "observaste el mundo."
        ),
    )


PLANNING_PROMPT_TEMPLATE = """
Tu objetivo es: {user_message}

Todavía no observaste la sala: no conocés los IDs exactos de los objetos ni
la disposición real del entorno. Antes de actuar, proponé un plan ordenado
de pasos en lenguaje natural (sin nombres de tools ni IDs, porque todavía no
los conocés) para cumplir el objetivo descripto arriba, usando solo lo que
el mensaje ya te contó. Ejemplo de paso: "buscar la llave que abre el cofre
del mismo color". Si el objetivo tiene un orden obligatorio (por ejemplo,
conseguir algo antes de abrir una puerta que se sella), reflejalo en el
orden del plan.

No asumas qué objeto abre qué cerradura antes de observarlo: si el mensaje
no te dice explícitamente que una llave abre una cerradura puntual, no lo
inventes en el plan (por ejemplo, no des por hecho que la primera llave que
aparece abre la puerta de salida). En esos casos, el paso puede decir algo
como "usar cada llave en el objeto cuya descripción confirme que es
compatible", en vez de adivinar una combinación concreta.
""".strip()


def render_plan(plan: Plan) -> str:
    """Devuelve el plan como bloque de texto para el system prompt."""
    lines = ["Plan inicial (generado antes de observar el mundo):"]
    lines.extend(
        f"{index}. {step}" for index, step in enumerate(plan.steps, start=1)
    )
    lines.append(
        "Este plan puede tener detalles incompletos o equivocados porque se "
        "hizo sin observar el mundo real. Usalo como guía de orden y "
        "alcance, pero priorizá siempre lo que confirmen las tools por "
        "sobre el plan si hay una contradicción."
    )
    return "\n".join(lines)
