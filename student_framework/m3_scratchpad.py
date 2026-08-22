"""Scratchpad determinista para los escenarios M3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from mia_agents.types import AgentStep


_ID_PATTERN = re.compile(r"\[id: ([^\]]+)\]")
_LOCATION_PATTERN = re.compile(r"^Estás en (.+)\.$", re.MULTILINE)
_ARRIVAL_PATTERN = re.compile(r"Llegas a (.+)\.$")
_EXITS_PATTERN = re.compile(r"^Salidas: (.+)\.$", re.MULTILINE)
_CARRIED_PATTERN = re.compile(r"^Llevas: (.+)\.$", re.MULTILINE)


@dataclass
class M3Scratchpad:
    """Estado de trabajo extraído de observaciones ya realizadas."""

    location: str | None = None
    exits: list[str] = field(default_factory=list)
    inventory: list[str] = field(default_factory=list)
    visible_ids: list[str] = field(default_factory=list)
    contents: dict[str, list[str]] = field(default_factory=dict)
    opened: list[str] = field(default_factory=list)
    examined: list[str] = field(default_factory=list)
    recent_errors: list[str] = field(default_factory=list)
    research_examined_ids: list[str] = field(default_factory=list)
    research_findings: list[str] = field(default_factory=list)

    def record(self, step: AgentStep) -> None:
        """Incorpora un resultado de tool sin inventar información nueva."""
        tool_name = step.tool_name or ""
        arguments = self._parse_arguments(step.tool_input)
        output = step.tool_output or ""

        if step.error is not None or output.startswith("Error:"):
            self._add_error(tool_name, arguments, output or step.error or "")
            return

        if tool_name == "look":
            self._record_look(output)
        elif tool_name == "go":
            self._record_go(output)
        elif tool_name == "take":
            self._record_take(arguments)
        elif tool_name == "examine":
            self._record_examine(arguments, output)
        elif tool_name == "use":
            self._record_use(arguments, output)
        elif tool_name == "research_documents":
            self._record_research(output)

    def render(self) -> str:
        """Devuelve una versión breve para sumar al system prompt dinámico."""
        lines = ["Scratchpad M3 (hechos observados mediante tools):"]

        if self.location:
            lines.append(f"- Ubicación actual: {self.location}")
        if self.exits:
            lines.append("- Salidas conocidas: " + ", ".join(self.exits))
        if self.inventory:
            lines.append("- Inventario: " + ", ".join(self.inventory))
        if self.visible_ids:
            lines.append("- IDs visibles: " + ", ".join(self.visible_ids))
        if self.contents:
            lines.append("- Contenido descubierto:")
            for target, item_ids in self.contents.items():
                rendered_items = ", ".join(item_ids) if item_ids else "sin IDs nuevos"
                lines.append(f"  - {target}: {rendered_items}")
        if self.opened:
            lines.append("- Objetos abiertos: " + ", ".join(self.opened))
        if self.research_examined_ids:
            lines.append(
                "- Documentos investigados: "
                + ", ".join(self.research_examined_ids)
            )
        if self.research_findings:
            lines.append("- Hallazgos de investigacion delegada:")
            lines.extend(f"  - {finding}" for finding in self.research_findings)
        if self.examined:
            lines.append("- Objetos examinados: " + ", ".join(self.examined))
        if self.recent_errors:
            lines.append("- Errores recientes:")
            lines.extend(f"  - {error}" for error in self.recent_errors)

        if len(lines) == 1:
            lines.append("- Sin hechos todavía. Observá el entorno antes de actuar.")

        return "\n".join(lines)

    def _record_look(self, output: str) -> None:
        location = _LOCATION_PATTERN.search(output)
        if location:
            self.location = location.group(1)

        exits = _EXITS_PATTERN.search(output)
        self.exits = self._split_csv(exits.group(1)) if exits else []

        self.visible_ids = self._unique(_ID_PATTERN.findall(output))

        carried = _CARRIED_PATTERN.search(output)
        if carried:
            self.inventory = self._unique(_ID_PATTERN.findall(carried.group(1)))

    def _record_go(self, output: str) -> None:
        arrival = _ARRIVAL_PATTERN.search(output)
        if arrival:
            self.location = arrival.group(1)
            self.exits = []
            self.visible_ids = []

    def _record_take(self, arguments: dict[str, object]) -> None:
        item = arguments.get("item")
        if isinstance(item, str) and item not in self.inventory:
            self.inventory.append(item)
            self.visible_ids = [known for known in self.visible_ids if known != item]

    def _record_examine(
        self,
        arguments: dict[str, object],
        output: str,
    ) -> None:
        target = arguments.get("target")
        if not isinstance(target, str):
            return

        if target not in self.examined:
            self.examined.append(target)

        discovered_ids = self._unique(_ID_PATTERN.findall(output))
        if discovered_ids:
            self.contents[target] = discovered_ids

    def _record_research(self, output: str) -> None:
        """Conserva hallazgos compactos de research_documents."""
        try:
            report = json.loads(output)
        except json.JSONDecodeError:
            return

        if not isinstance(report, dict):
            return

        examined_ids = report.get("examined_document_ids", [])
        if isinstance(examined_ids, list):
            self.research_examined_ids = self._unique(
                self.research_examined_ids
                + [
                    document_id
                    for document_id in examined_ids
                    if isinstance(document_id, str)
                ]
            )

        findings = report.get("findings", [])
        if not isinstance(findings, list):
            return

        for finding in findings:
            if not isinstance(finding, dict):
                continue

            source_id = finding.get("source_id")
            if not isinstance(source_id, str):
                continue

            revealed_items = finding.get("revealed_items", [])
            item_ids = [
                item.get("id")
                for item in revealed_items
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
            relevant_fact = finding.get("relevant_fact")
            parts = [source_id]
            if item_ids:
                parts.append("revela " + ", ".join(item_ids))
            if isinstance(relevant_fact, str) and relevant_fact:
                parts.append(relevant_fact)

            record = ": ".join([parts[0], "; ".join(parts[1:])])
            if record not in self.research_findings:
                self.research_findings.append(record)

    def _record_use(
        self,
        arguments: dict[str, object],
        output: str,
    ) -> None:
        target = arguments.get("target")

        if "no encaja" in output or "no pasa nada" in output:
            self._add_error("use", arguments, output)
            return

        if (
            isinstance(target, str)
            and "se abre" in output.lower()
            and target not in self.opened
        ):
            self.opened.append(target)

    def _add_error(
        self,
        tool_name: str,
        arguments: dict[str, object],
        output: str,
    ) -> None:
        compact_output = " ".join(output.split())[:220]
        record = f"{tool_name}({arguments}): {compact_output}"
        if record not in self.recent_errors:
            self.recent_errors.append(record)
        self.recent_errors = self.recent_errors[-5:]

    @staticmethod
    def _parse_arguments(raw_arguments: str | None) -> dict[str, object]:
        try:
            parsed = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _split_csv(value: str) -> list[str]:
        return [part.strip() for part in value.split(",") if part.strip()]

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))
