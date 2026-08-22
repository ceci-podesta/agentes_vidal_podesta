"""Investigacion delegada y acotada para escenarios M3.

La tool principal observa cada documento solicitado y delega solamente la
sintesis al LLM. Asi el worker no puede omitir documentos ni devolver JSON
libre sin validacion: structured_call_with_usage valida ResearchReport.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field

from mia_agents.protocols import LLMClient
from mia_agents.types import ToolSchema

from .agent import (
    MyAgent,
    StructuredCallError,
    StructuredCallUsage,
)


MAX_DOCUMENTS_PER_REQUEST = 20
WORKER_BATCH_SIZE = 5
MAX_RELEVANT_FACT_LENGTH = 280

_CONTAINED_ITEM_PATTERN = re.compile(
    r"^\s*-\s*(.*?)\s+\[id:\s*([^\]]+)\]",
    re.MULTILINE,
)


class RevealedItem(BaseModel):
    """Objeto que una observacion revelo dentro de un documento."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class ResearchFinding(BaseModel):
    """Hallazgo compacto asociado a un documento inspeccionado."""

    source_id: str = Field(min_length=1)
    revealed_items: list[RevealedItem] = Field(default_factory=list)
    relevant_fact: str = Field(
        default="",
        max_length=MAX_RELEVANT_FACT_LENGTH,
    )


class ResearchReport(BaseModel):
    """Contrato compacto entre la investigacion y el agente principal."""

    examined_document_ids: list[str] = Field(default_factory=list)
    findings: list[ResearchFinding] = Field(default_factory=list)
    worker_errors: list[str] = Field(default_factory=list)


@dataclass
class DocumentObservation:
    """Resultado de observar un documento dentro de un lote autorizado."""

    document_id: str
    output: str
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and not self.output.startswith("Error:")


@dataclass
class ResearchDiagnostics:
    """Metricas de workers que no viven en AgentResult del principal."""

    workers_started: int = 0
    worker_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    has_token_usage: bool = False
    worker_errors: list[str] = field(default_factory=list)

    def record_worker_usage(
        self,
        *,
        observation_count: int,
        usage: StructuredCallUsage | None,
        errors: list[str],
    ) -> None:
        """Registra observaciones y uso del sintetizador de un lote."""
        self.workers_started += 1
        self.worker_tool_calls += observation_count

        if usage is not None:
            if usage.input_tokens is not None or usage.output_tokens is not None:
                self.has_token_usage = True
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0

        for error in errors:
            self.record_error(error)

    def record_error(self, error: str) -> None:
        if error not in self.worker_errors:
            self.worker_errors.append(error)

    def as_dict(self) -> dict[str, Any]:
        return {
            "workers_started": self.workers_started,
            "worker_tool_calls": self.worker_tool_calls,
            "input_tokens": (
                self.input_tokens if self.has_token_usage else None
            ),
            "output_tokens": (
                self.output_tokens if self.has_token_usage else None
            ),
            "worker_errors": list(self.worker_errors),
        }


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _observe_batch(
    examine_tool: Callable[..., str],
    document_ids: list[str],
) -> list[DocumentObservation]:
    """Inspecciona cada ID del lote una vez y conserva su resultado."""
    observations: list[DocumentObservation] = []

    for document_id in document_ids:
        try:
            output = examine_tool(target=document_id)
        except Exception as exc:
            error = f"No se pudo inspeccionar {document_id}: {exc}"
            observations.append(
                DocumentObservation(
                    document_id=document_id,
                    output=f"Error: {error}",
                    error=error,
                )
            )
            continue

        output_str = output if isinstance(output, str) else str(output)
        error = output_str.removeprefix("Error: ").strip()
        observations.append(
            DocumentObservation(
                document_id=document_id,
                output=output_str,
                error=error if output_str.startswith("Error:") else None,
            )
        )

    return observations


def _actual_examined_ids(
    observations: list[DocumentObservation],
) -> list[str]:
    """Devuelve solamente las observaciones exitosas del lote."""
    return _unique(
        [
            observation.document_id
            for observation in observations
            if observation.succeeded
        ]
    )


def _observation_errors(
    observations: list[DocumentObservation],
) -> list[str]:
    return [
        observation.error
        for observation in observations
        if observation.error is not None
    ]


def _fallback_findings(
    observations: list[DocumentObservation],
) -> list[ResearchFinding]:
    """Preserva objetos revelados si falla toda la sintesis estructurada."""
    findings: list[ResearchFinding] = []

    for observation in observations:
        if not observation.succeeded or "Contiene:" not in observation.output:
            continue

        contained_text = observation.output.split("Contiene:", maxsplit=1)[1]
        revealed_items = [
            RevealedItem(id=item_id.strip(), name=name.strip())
            for name, item_id in _CONTAINED_ITEM_PATTERN.findall(contained_text)
        ]
        if revealed_items:
            findings.append(
                ResearchFinding(
                    source_id=observation.document_id,
                    revealed_items=revealed_items,
                    relevant_fact="",
                )
            )

    return findings



def _replace_revealed_items_with_verified(
    report: ResearchReport,
    observations: list[DocumentObservation],
) -> ResearchReport:
    """Reemplaza IDs inventados por objetos realmente revelados por examine."""
    verified_by_source = {
        finding.source_id: finding.revealed_items
        for finding in _fallback_findings(observations)
    }
    errors: list[str] = []
    relevant_facts_by_source: dict[str, str] = {}
    source_order: list[str] = []

    for finding in report.findings:
        if finding.source_id not in source_order:
            source_order.append(finding.source_id)
            relevant_facts_by_source[finding.source_id] = (
                finding.relevant_fact
            )

        verified_item_ids = {
            item.id
            for item in verified_by_source.get(finding.source_id, [])
        }
        claimed_item_ids = {item.id for item in finding.revealed_items}
        invented_item_ids = claimed_item_ids - verified_item_ids
        if invented_item_ids:
            errors.append(
                "El worker reporto objetos no revelados por examine en "
                f"{finding.source_id}: "
                + ", ".join(sorted(invented_item_ids))
            )

    for source_id in verified_by_source:
        if source_id not in source_order:
            source_order.append(source_id)

    return ResearchReport(
        examined_document_ids=report.examined_document_ids,
        findings=[
            ResearchFinding(
                source_id=source_id,
                revealed_items=verified_by_source.get(source_id, []),
                relevant_fact=relevant_facts_by_source.get(source_id, ""),
            )
            for source_id in source_order
        ],
        worker_errors=report.worker_errors + errors,
    )
def _normalize_report(
    report: ResearchReport,
    actual_examined_ids: list[str],
) -> tuple[ResearchReport, list[str]]:
    """Descarta afirmaciones del sintetizador sin un origen observado."""
    actual_ids = set(actual_examined_ids)
    errors: list[str] = []
    valid_findings: list[ResearchFinding] = []

    claimed_ids = set(report.examined_document_ids)
    unobserved_claims = claimed_ids - actual_ids
    if unobserved_claims:
        errors.append(
            "El worker declaro documentos no inspeccionados: "
            + ", ".join(sorted(unobserved_claims))
        )

    for finding in report.findings:
        if finding.source_id not in actual_ids:
            errors.append(
                "El worker reporto un hallazgo sin inspeccionar su origen: "
                + finding.source_id
            )
            continue
        valid_findings.append(finding)

    return (
        ResearchReport(
            examined_document_ids=actual_examined_ids,
            findings=valid_findings,
            worker_errors=report.worker_errors + errors,
        ),
        errors,
    )


def _synthesis_prompt(
    observations: list[DocumentObservation],
    objective: str,
) -> str:
    """Arma el unico contexto que recibe el LLM sintetizador del lote."""
    observed_documents = [
        {
            "document_id": observation.document_id,
            "content": observation.output,
            "error": observation.error,
        }
        for observation in observations
    ]
    documents_json = json.dumps(observed_documents, ensure_ascii=False)

    return f"""Objetivo del agente principal:
{objective}

Sos un worker sintetizador. A continuacion recibis observaciones ya realizadas
sobre un lote cerrado de documentos. No podes pedir ni asumir observaciones
adicionales.

{documents_json}

Devolve un ResearchReport usando final_result. Inclui los IDs realmente
inspeccionados, objetos revelados y una oracion breve para cada pista relevante
al objetivo. No copies prosa extensa. Si no hay hallazgos, usa findings vacio.
No inventes IDs, contenido ni errores."""


def _run_worker(
    *,
    llm_client: LLMClient,
    examine_tool: Callable[..., str],
    document_ids: list[str],
    objective: str,
    diagnostics: ResearchDiagnostics,
) -> ResearchReport:
    """Observa un lote y delega solamente su sintesis semantica al LLM."""
    observations = _observe_batch(examine_tool, document_ids)
    actual_examined_ids = _actual_examined_ids(observations)
    observation_errors = _observation_errors(observations)

    synthesizer = MyAgent(
        llm_client=llm_client,
        system_prompt=(
            "Sos un sintetizador de investigacion acotada. "
            "No inventes hechos ni IDs."
        ),
        max_iterations=1,
        max_history_messages=20,
    )

    try:
        structured_result = synthesizer.structured_call_with_usage(
            prompt=_synthesis_prompt(observations, objective),
            schema=ResearchReport,
        )
    except StructuredCallError as exc:
        error = f"Reporte estructurado invalido del worker: {exc}"
        all_errors = observation_errors + [error]
        diagnostics.record_worker_usage(
            observation_count=len(observations),
            usage=exc.usage,
            errors=all_errors,
        )
        return ResearchReport(
            examined_document_ids=actual_examined_ids,
            findings=_fallback_findings(observations),
            worker_errors=all_errors,
        )

    normalized_report, _ = _normalize_report(
        structured_result.value,
        actual_examined_ids,
    )
    normalized_report = _replace_revealed_items_with_verified(
        normalized_report,
        observations,
    )
    all_errors = observation_errors + normalized_report.worker_errors
    diagnostics.record_worker_usage(
        observation_count=len(observations),
        usage=structured_result.usage,
        errors=all_errors,
    )

    return ResearchReport(
        examined_document_ids=normalized_report.examined_document_ids,
        findings=normalized_report.findings,
        worker_errors=all_errors,
    )


def make_research_documents_tool(
    *,
    llm_client: LLMClient,
    examine_tool: Callable[..., str],
    examine_schema: ToolSchema,
    diagnostics: ResearchDiagnostics,
) -> tuple[Callable[..., str], ToolSchema]:
    """Crea una tool que investiga una coleccion completa por lotes internos.

    examine_schema se conserva en la firma para no romper el runner existente.
    Ya no se registra en un worker: las observaciones se ejecutan antes de la
    sintesis estructurada.
    """
    _ = examine_schema

    def research_documents(
        document_ids: Annotated[
            list[str],
            Field(
                description=(
                    "IDs de todos los documentos de la coleccion a investigar. "
                    "La tool los divide internamente en lotes pequenos."
                ),
                min_length=1,
                max_length=MAX_DOCUMENTS_PER_REQUEST,
            ),
        ],
        objective: Annotated[
            str,
            Field(
                description=(
                    "Objetivo actual que permite distinguir hallazgos "
                    "relevantes de informacion decorativa."
                )
            ),
        ],
    ) -> str:
        """Investiga documentos y devuelve hallazgos compactos con IDs."""
        if not isinstance(document_ids, list) or not all(
            isinstance(document_id, str) and document_id
            for document_id in document_ids
        ):
            return "Error: document_ids debe ser una lista no vacia de IDs."

        unique_document_ids = _unique(document_ids)
        if len(unique_document_ids) > MAX_DOCUMENTS_PER_REQUEST:
            return (
                "Error: la investigacion admite como maximo "
                f"{MAX_DOCUMENTS_PER_REQUEST} documentos por llamada."
            )

        if not isinstance(objective, str) or not objective.strip():
            return "Error: objective debe ser un texto no vacio."

        reports: list[ResearchReport] = []
        for start in range(0, len(unique_document_ids), WORKER_BATCH_SIZE):
            batch = unique_document_ids[start : start + WORKER_BATCH_SIZE]
            reports.append(
                _run_worker(
                    llm_client=llm_client,
                    examine_tool=examine_tool,
                    document_ids=batch,
                    objective=objective,
                    diagnostics=diagnostics,
                )
            )

        combined_report = ResearchReport(
            examined_document_ids=_unique(
                [
                    document_id
                    for report in reports
                    for document_id in report.examined_document_ids
                ]
            ),
            findings=[
                finding
                for report in reports
                for finding in report.findings
            ],
            worker_errors=[
                error
                for report in reports
                for error in report.worker_errors
            ],
        )
        return json.dumps(combined_report.model_dump(), ensure_ascii=False)

    return research_documents, ToolSchema.from_callable(research_documents)
