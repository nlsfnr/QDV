#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Optional

import click

sys.path.append(".")

import qdv  # noqa: E402

_SAVE_PATH = Path("tmp/coach-example/")


@click.group()
def cli() -> None:
    pass


# fmt: off
@cli.command("new")
@click.option("--openai-key-path", "-k", type=Path, required=False,
              help="Path to the file containing the OpenAI API key")
# fmt: on
def new(
    openai_key_path: Optional[Path],
) -> None:
    (
        qdv.OpenAIE2ESolution.from_item_store(
            _load_text_store(), api_key=openai_key_path
        ).save(_SAVE_PATH)
    )


# fmt: off
@cli.command("query")
@click.option("--openai-key-path", "-k", type=Path, required=False,
              help="Path to the file containing the OpenAI API key")
@click.option("--query", "-q", type=str, required=True, help="Query text")
# fmt: on
def query(
    openai_key_path: Optional[Path],
    query: str,
) -> None:
    print(
        qdv.OpenAIE2ESolution.load(_SAVE_PATH, api_key=openai_key_path).query(
            query, top_k=5
        )
    )


def _load_text_store() -> qdv.JsonStore:
    with open(Path(__file__) / "coach-data.json", "r") as f:
        dicts = json.load(f)
    data = {d["id"]: d["description"] for d in dicts}
    assert all(isinstance(k, str) for k in data.keys())
    assert all(isinstance(v, str) for v in data.values())
    return qdv.JsonStore(data)


if __name__ == "__main__":
    cli()
