"""Tests para investigacion delegada con observacion y sintesis separadas."""

from __future__ import annotations

import json

from mia_agents.testing import MockLLMClient
from mia_agents.tool_schema import FINAL_RESULT_TOOL_NAME
from mia_agents.types import AgentStep, LLMResponse, ToolCall, ToolSchema
from student_framework.m3_research import (
    ResearchDiagnostics,
    make_research_documents_tool,
)
from student_framework.m3_scratchpad import M3Scratchpad


def _examine_schema() -> ToolSchema:
    def examine(target: str) -> str:
        """Inspecciona un documento por ID."""
        return target

    return ToolSchema.from_callable(examine)


def _final_result_response(
    report: dict[str, object],
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="final-1",
                name=FINAL_RESULT_TOOL_NAME,
                arguments=json.dumps(report),
            )
        ],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_research_observes_batch_and_returns_structured_findings() -> None:
    calls: list[str] = []

    def examine(target: str) -> str:
        calls.append(target)
        if target == "expediente_b":
            return (
                "expediente b: texto largo.\n"
                "Contiene:\n"
                "  - llave del archivo [id: llave_archivo]"
            )
        return "expediente a: texto largo sin hallazgos."

    mock = MockLLMClient(
        [
            _final_result_response(
                {
                    "examined_document_ids": [
                        "expediente_a",
                        "expediente_b",
                    ],
                    "findings": [
                        {
                            "source_id": "expediente_b",
                            "revealed_items": [
                                {
                                    "id": "llave_archivo",
                                    "name": "llave del archivo",
                                }
                            ],
                            "relevant_fact": (
                                "Revela una llave institucional util "
                                "para abrir la puerta."
                            ),
                        }
                    ],
                },
                input_tokens=14,
                output_tokens=4,
            )
        ]
    )
    diagnostics = ResearchDiagnostics()
    tool, schema = make_research_documents_tool(
        llm_client=mock,
        examine_tool=examine,
        examine_schema=_examine_schema(),
        diagnostics=diagnostics,
    )

    report = json.loads(
        tool(
            document_ids=["expediente_a", "expediente_b"],
            objective="Encontrar la forma de abrir la puerta.",
        )
    )

    assert schema.name == "research_documents"
    assert calls == ["expediente_a", "expediente_b"]
    assert report["examined_document_ids"] == [
        "expediente_a",
        "expediente_b",
    ]
    assert report["findings"][0]["source_id"] == "expediente_b"
    assert report["findings"][0]["revealed_items"] == [
        {"id": "llave_archivo", "name": "llave del archivo"}
    ]

    metrics = diagnostics.as_dict()
    assert metrics["workers_started"] == 1
    assert metrics["worker_tool_calls"] == 2
    assert metrics["input_tokens"] == 14
    assert metrics["output_tokens"] == 4

    assert len(mock.calls) == 1
    assert [tool_schema.name for tool_schema in mock.calls[0]["tools"]] == [
        FINAL_RESULT_TOOL_NAME
    ]
    assert "expediente_a" in str(mock.calls[0]["messages"])
    assert "expediente_b" in str(mock.calls[0]["messages"])


def test_research_discards_hallucinated_document_claims() -> None:
    calls: list[str] = []

    def examine(target: str) -> str:
        calls.append(target)
        return "Documento sin pista."

    mock = MockLLMClient(
        [
            _final_result_response(
                {
                    "examined_document_ids": ["fuera_del_lote"],
                    "findings": [
                        {
                            "source_id": "fuera_del_lote",
                            "revealed_items": [],
                            "relevant_fact": "Dato inventado.",
                        }
                    ],
                }
            )
        ]
    )
    diagnostics = ResearchDiagnostics()
    tool, _ = make_research_documents_tool(
        llm_client=mock,
        examine_tool=examine,
        examine_schema=_examine_schema(),
        diagnostics=diagnostics,
    )

    report = json.loads(
        tool(
            document_ids=["expediente_a"],
            objective="Encontrar una pista.",
        )
    )

    assert calls == ["expediente_a"]
    assert report["examined_document_ids"] == ["expediente_a"]
    assert report["findings"] == []
    assert report["worker_errors"]


def test_research_keeps_revealed_items_when_synthesis_exhausts_repairs() -> None:
    def examine(target: str) -> str:
        return (
            "expediente: texto largo.\n"
            "Contiene:\n"
            "  - llave del archivo [id: llave_archivo]"
        )

    mock = MockLLMClient(
        [
            LLMResponse(content="texto libre", input_tokens=10, output_tokens=1),
            LLMResponse(content="texto libre", input_tokens=20, output_tokens=2),
            LLMResponse(content="texto libre", input_tokens=30, output_tokens=3),
        ]
    )
    diagnostics = ResearchDiagnostics()
    tool, _ = make_research_documents_tool(
        llm_client=mock,
        examine_tool=examine,
        examine_schema=_examine_schema(),
        diagnostics=diagnostics,
    )

    report = json.loads(
        tool(
            document_ids=["expediente_a"],
            objective="Encontrar una llave.",
        )
    )

    assert report["examined_document_ids"] == ["expediente_a"]
    assert report["findings"] == [
        {
            "source_id": "expediente_a",
            "revealed_items": [
                {"id": "llave_archivo", "name": "llave del archivo"}
            ],
            "relevant_fact": "",
        }
    ]
    assert report["worker_errors"]

    metrics = diagnostics.as_dict()
    assert metrics["workers_started"] == 1
    assert metrics["worker_tool_calls"] == 1
    assert metrics["input_tokens"] == 60
    assert metrics["output_tokens"] == 6


def test_scratchpad_keeps_delegated_findings() -> None:
    scratchpad = M3Scratchpad()
    scratchpad.record(
        AgentStep(
            tool_name="research_documents",
            tool_input="{}",
            tool_output=json.dumps(
                {
                    "examined_document_ids": ["expediente_7240"],
                    "findings": [
                        {
                            "source_id": "expediente_7240",
                            "revealed_items": [
                                {
                                    "id": "llave_archivo",
                                    "name": "llave del archivo",
                                }
                            ],
                            "relevant_fact": "Contiene una llave util.",
                        }
                    ],
                }
            ),
        )
    )

    rendered = scratchpad.render()

    assert "Documentos investigados: expediente_7240" in rendered
    assert "expediente_7240: revela llave_archivo" in rendered

def test_research_replaces_invented_items_with_verified_revealed_items() -> None:
    """Los objetos del reporte deben provenir del output real de examine."""

    def examine(target: str) -> str:
        return (
            "expediente: texto largo.\n"
            "Contiene:\n"
            "  - llave real [id: llave_real]"
        )

    mock = MockLLMClient(
        [
            _final_result_response(
                {
                    "examined_document_ids": ["expediente_a"],
                    "findings": [
                        {
                            "source_id": "expediente_a",
                            "revealed_items": [
                                {
                                    "id": "id_inventado",
                                    "name": "objeto inventado",
                                }
                            ],
                            "relevant_fact": "El documento contiene una pista.",
                        }
                    ],
                }
            )
        ]
    )
    diagnostics = ResearchDiagnostics()
    tool, _ = make_research_documents_tool(
        llm_client=mock,
        examine_tool=examine,
        examine_schema=_examine_schema(),
        diagnostics=diagnostics,
    )

    report = json.loads(
        tool(
            document_ids=["expediente_a"],
            objective="Encontrar una llave.",
        )
    )

    assert report["findings"] == [
        {
            "source_id": "expediente_a",
            "revealed_items": [
                {"id": "llave_real", "name": "llave real"}
            ],
            "relevant_fact": "El documento contiene una pista.",
        }
    ]
    assert report["worker_errors"]


def test_repeated_research_call_is_blocked_by_observation_guard() -> None:
    """El agente no vuelve a ejecutar la misma investigación sin novedad."""

    from student_framework.agent import MyAgent

    calls: list[dict[str, object]] = []

    def research_documents(
        document_ids: list[str],
        objective: str,
    ) -> str:
        calls.append(
            {
                "document_ids": document_ids,
                "objective": objective,
            }
        )
        return '{"findings": []}'

    schema = ToolSchema.from_callable(research_documents)
    arguments = json.dumps(
        {
            "document_ids": ["expediente_a", "expediente_b"],
            "objective": "Encontrar la llave.",
        }
    )
    mock = MockLLMClient(
        [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="research-1",
                        name=schema.name,
                        arguments=arguments,
                    )
                ],
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="research-2",
                        name=schema.name,
                        arguments=arguments,
                    )
                ],
            ),
            LLMResponse(content="No hay más acciones útiles."),
        ]
    )
    agent = MyAgent(
        llm_client=mock,
        max_iterations=3,
        max_repeated_observations=1,
        observation_tool_names={schema.name},
    )
    agent.register_tool(research_documents, schema)

    result = agent.run("Investigá los documentos.")

    assert calls == [
        {
            "document_ids": ["expediente_a", "expediente_b"],
            "objective": "Encontrar la llave.",
        }
    ]
    assert len(result.steps) == 2
    assert result.steps[1].error is not None
    assert "observ" in result.steps[1].error.lower()

