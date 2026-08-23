from eval.error_categories import categorize_report, categorize_result


def _step(tool_name: str, tool_input: str, tool_output: str, error: str | None = None) -> dict[str, object]:
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "error": error,
    }


def test_flags_id_inventado() -> None:
    result = {
        "scenario": "case",
        "difficulty": "easy",
        "goal_achieved": False,
        "goal_reason": "puerta_principal está cerrada",
        "agent_result": {
            "steps": [
                _step("examine", "target=llave_x", "Error: no existe ningún objeto con id 'llave_x'."),
            ],
            "error": "Se alcanzó el máximo de iteraciones (10) sin que el modelo produjera una respuesta final.",
        },
    }

    categories = categorize_result(result)["categories"]

    assert "id_inventado" in categories
    assert "limite_iteraciones" in categories


def test_flags_violacion_de_estado() -> None:
    result = {
        "scenario": "case",
        "difficulty": "medium",
        "goal_achieved": False,
        "goal_reason": "puerta_principal está cerrada",
        "agent_result": {
            "steps": [
                _step("use", "item=llave, target=puerta_principal", "Error: no llevas ningún 'llave'."),
            ],
            "error": None,
        },
    }

    categories = categorize_result(result)["categories"]

    assert "violacion_estado" in categories
    assert "terminacion_prematura" in categories


def test_flags_accion_redundante_por_repeticion_consecutiva() -> None:
    result = {
        "scenario": "case",
        "difficulty": "medium",
        "goal_achieved": True,
        "goal_reason": "puerta_principal está abierta",
        "agent_result": {
            "steps": [
                _step("look", "", "Ves una sala."),
                _step("look", "", "Ves una sala."),
            ],
            "error": None,
        },
    }

    categories = categorize_result(result)["categories"]

    assert "accion_redundante" in categories


def test_flags_accion_redundante_por_guardia_del_propio_agente() -> None:
    result = {
        "scenario": "apartment-keys",
        "difficulty": "medium",
        "goal_achieved": True,
        "goal_reason": "puerta_principal está abierta",
        "agent_result": {
            "steps": [
                _step(
                    "look",
                    "{}",
                    "Error: La observación ya fue realizada sin que hubiera "
                    "progreso. Elegí una acción diferente.",
                    error=(
                        "La observación ya fue realizada sin que hubiera "
                        "progreso. Elegí una acción diferente."
                    ),
                ),
                _step(
                    "go",
                    '{"direction": "norte"}',
                    "Error: La llamada ya falló sin que hubiera progreso. "
                    "Elegí una acción diferente.",
                    error=(
                        "La llamada ya falló sin que hubiera progreso. "
                        "Elegí una acción diferente."
                    ),
                ),
            ],
            "error": None,
        },
    }

    categories = categorize_result(result)["categories"]

    assert len(categories["accion_redundante"]) == 2


def test_flags_planificacion_orden_incorrecto_en_goal_de_secuencia() -> None:
    result = {
        "scenario": "office-sequence",
        "difficulty": "hard",
        "goal_achieved": False,
        "goal_reason": "el evento 'take:documento_confidencial' no ocurrió en el orden requerido",
        "agent_result": {"steps": [], "error": None},
    }

    categories = categorize_result(result)["categories"]

    assert "planificacion_orden_incorrecto" in categories


def test_flags_presion_de_contexto_por_tokens_altos_sin_lograr_el_goal() -> None:
    result = {
        "scenario": "extreme-archive",
        "difficulty": "extreme",
        "goal_achieved": False,
        "goal_reason": "puerta_principal está cerrada",
        "agent_result": {
            "steps": [],
            "error": "Se alcanzó el máximo de iteraciones (25) sin que el modelo produjera una respuesta final.",
            "input_tokens": 120_000,
        },
    }

    categories = categorize_result(result)["categories"]

    assert "presion_contexto" in categories
    assert "limite_iteraciones" in categories


def test_goal_achieved_no_marca_terminacion_prematura_ni_orden_incorrecto() -> None:
    result = {
        "scenario": "case",
        "difficulty": "easy",
        "goal_achieved": True,
        "goal_reason": "puerta_principal está abierta",
        "agent_result": {"steps": [], "error": None},
    }

    categories = categorize_result(result)["categories"]

    assert "terminacion_prematura" not in categories
    assert "planificacion_orden_incorrecto" not in categories


def test_categorize_report_agrega_totales_por_categoria() -> None:
    report = {
        "results": [
            {
                "scenario": "a",
                "difficulty": "easy",
                "goal_achieved": True,
                "goal_reason": "puerta_principal está abierta",
                "agent_result": {"steps": [], "error": None},
            },
            {
                "scenario": "b",
                "difficulty": "hard",
                "goal_achieved": False,
                "goal_reason": "puerta_principal está cerrada",
                "agent_result": {
                    "steps": [
                        _step("use", "item=x, target=y", "Error: no existe ningún objeto con id 'x'."),
                    ],
                    "error": None,
                },
            },
        ]
    }

    analysis = categorize_report(report)

    assert analysis["totals"]["id_inventado"] == 1
    assert analysis["totals"]["terminacion_prematura"] == 1
    assert len(analysis["per_scenario"]) == 2
