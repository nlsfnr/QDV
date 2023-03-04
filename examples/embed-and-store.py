#!/usr/bin/env python3
import click
from pathlib import Path
import sys
sys.path.append('.')

import vdb


@click.group()
def cli() -> None:
    pass


@cli.command('add')
@click.argument('id', type=str)
@click.argument('text', type=str)
@click.option('--path', '-p', type=Path, default=Path('./tmp/embed-and-store/'))
def cli_add(id: str,
            text: str,
            path: Path,
            ) -> None:
    embedder = vdb.CLIPTextEmbedder()
    store = vdb.LMDBStore(path,
                          embedding_dim=embedder.dim)
    store.store([id], embedder([text]))


@cli.command('query')
@click.argument('text', type=str)
@click.option('--path', '-p', type=Path, default=Path('./tmp/embed-and-store/'))
@click.option('--topk', '-k', type=int, default=5)
def cli_query(text: str,
              path: Path,
              topk: int,
              ) -> None:
    embedder = vdb.CLIPTextEmbedder()
    store = vdb.LMDBStore(path,
                          embedding_dim=embedder.dim)
    index = vdb.BruteForceIndex(store)
    ids, distances = index.query(embedder([text]), topk)
    print(ids, distances)


if __name__ == '__main__':
    cli()
