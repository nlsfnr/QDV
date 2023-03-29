#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from typing import Dict, List

import click

sys.path.append(".")

import qdv  # noqa: E402


def load_data(path: Path) -> List[Dict[str, str]]:
    with open(path, "r") as f:
        reader = csv.reader(f)
        entries = list(reader)
        dicts = [dict(zip(entries[0], row)) for row in entries[1:]]
    return dicts


@click.group()
def cli() -> None:
    pass


# fmt: off
@cli.command("new")
@click.option("--openai-key-path", "-k", type=Path, required=True,
              help="Path to the file containing the OpenAI API key")
# fmt: on
def new(
    openai_key_path: Path,
) -> None:
    path = Path("tmp/coach-example/")
    with openai_key_path.open("r") as f:
        key = f.read().strip()

    # Load the data
    data = load_data(Path(__file__).parent / "coach-data.csv")
    ids = [d["ID"] for d in data]
    texts = [d["Description"] for d in data]

    store_path = path / "store"
    did_exist = store_path.exists()

    embedder = qdv.OpenAIEmbedder(key)
    store = qdv.LMDBStore(store_path, embedding_dim=embedder.dim)

    if not did_exist:
        embeddings = embedder(texts)
        store.store(ids, embeddings)

    assert len(store) == len(ids), f"Expected {len(ids)} embeddings, got {len(store)}"

    # Create an index
    index = qdv.KNNIndex.from_store(store)
    index.save(path / "index")


# fmt: off
@cli.command("search")
@click.option("--openai-key-path", "-k", type=Path, required=True,
              help="Path to the file containing the OpenAI API key")
@click.option("--query", "-q", type=str, required=True, help="Query text")
# fmt: on
def search(
    openai_key_path: Path,
    query: str,
) -> None:
    path = Path("tmp/coach-example/")
    with openai_key_path.open("r") as f:
        key = f.read().strip()

    # Load the data, embedder and index
    data = load_data(Path(__file__).parent / "coach-data.csv")
    embedder = qdv.OpenAIEmbedder(key)
    index = qdv.KNNIndex.load(path / "index")

    query_embeddings = embedder([query])
    ids, distances = index.query(query_embeddings, k=5)

    print("Results:")
    for i, (id, distance) in enumerate(zip(ids[0], distances[0])):
        coach = [d["Name"] for d in data if d["ID"] == id][0]
        print(f"{i+1}. {coach} (distance: {distance:.4f})")


if __name__ == "__main__":
    cli()
