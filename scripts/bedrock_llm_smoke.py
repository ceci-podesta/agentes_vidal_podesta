#!/usr/bin/env python3
"""Prueba mínima de Bedrock usando el cliente LLM del andamiaje.

Configuren las credenciales por variables de entorno antes de ejecutar:

    export AWS_ACCESS_KEY_ID="<su_access_key>"
    export AWS_SECRET_ACCESS_KEY="<su_secret_key>"
    export AWS_SESSION_TOKEN="<su_session_token>"  # si usan credenciales temporales
    export AWS_REGION="us-east-1"
    export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"

Luego, desde la carpeta `scaffold/`:

    python scripts/bedrock_llm_smoke.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


RECOMMENDED_MODELS = {
    "amazon.nova-micro-v1:0": "Baseline débil para ablaciones",
    "amazon.nova-lite-v1:0": "Default recomendado",
    "amazon.nova-pro-v1:0": "Run fuerte, M3 extreme",
}


def _add_scaffold_to_path() -> None:
    scaffold_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(scaffold_dir))


def _missing_required_env() -> list[str]:
    required = ["BEDROCK_MODEL_ID"]
    return [name for name in required if not os.environ.get(name)]


def _credential_source_hint() -> str:
    if os.environ.get("AWS_PROFILE"):
        return f"AWS_PROFILE={os.environ['AWS_PROFILE']}"
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        has_session = "sí" if os.environ.get("AWS_SESSION_TOKEN") else "no"
        return (
            "variables AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
            f"(session token: {has_session})"
        )
    return "cadena estándar de boto3 (perfil local, rol, SSO o metadata)"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hace una llamada LLM simple a Bedrock usando mia_agents.llm_client.",
    )
    parser.add_argument(
        "--message",
        default="Respondé en una frase: ¿qué es un agente con herramientas?",
        help="Mensaje de usuario para enviar al modelo.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura de la llamada (defecto: 0.2).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _add_scaffold_to_path()

    from mia_agents._env import load_env_files
    from mia_agents.llm_client import BedrockProvider, LLMClient

    load_env_files()

    args = _build_parser().parse_args(argv)
    missing = _missing_required_env()
    if missing:
        print("Faltan variables de entorno requeridas:", file=sys.stderr)
        for name in missing:
            print(f"  - {name}", file=sys.stderr)
        print("\nEjemplo:", file=sys.stderr)
        print(
            '  export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"',
            file=sys.stderr,
        )
        print('  export AWS_REGION="us-east-1"', file=sys.stderr)
        print("\nModelos recomendados:", file=sys.stderr)
        for model_id, purpose in RECOMMENDED_MODELS.items():
            print(f"  - {model_id}: {purpose}", file=sys.stderr)
        return 2

    model_id = os.environ["BEDROCK_MODEL_ID"]
    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

    print("Configuración Bedrock")
    print(f"  Modelo      : {model_id}")
    if model_id in RECOMMENDED_MODELS:
        print(f"  Uso esperado: {RECOMMENDED_MODELS[model_id]}")
    print(f"  Región      : {region}")
    print(f"  Credenciales: {_credential_source_hint()}")
    print()

    client = LLMClient(BedrockProvider())
    response = client.chat(
        messages=[{"role": "user", "content": args.message}],
        system="Respondé en español rioplatense, de forma breve y clara.",
        temperature=args.temperature,
    )

    print("Respuesta")
    print(response.content or "(sin texto)")
    print()
    print("Metadatos")
    print(f"  input_tokens : {response.input_tokens}")
    print(f"  output_tokens: {response.output_tokens}")
    if response.raw_response:
        print(f"  stopReason   : {response.raw_response.get('stopReason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
