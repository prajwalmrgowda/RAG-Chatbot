"""Portable corpus snapshots, independent of Chroma's internal index files."""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from collections import defaultdict
from datetime import date
from pathlib import Path

from src.config import APPROVED_SOURCE_ROLES, SUPPORTED_SCHEMES
from src.models import Chunk
from src.store import ChromaStore, IndexConfig, StoreError

SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "data/corpus.json.gz"


def export_snapshot(store: ChromaStore, path: Path = SNAPSHOT_PATH) -> None:
    records = []
    for url in sorted(APPROVED_SOURCE_ROLES):
        result = store.get_source(url, include_embeddings=True)
        if not result['ids']:
            continue
        for identifier, document, metadata, vector in zip(
            result['ids'], result['documents'], result['metadatas'], result['embeddings']
        ):
            records.append({
                "id": identifier, "text": document, "metadata": metadata,
                "embedding": [float(value) for value in vector],
            })
    if not records or len(records) != store.count():
        raise StoreError("Snapshot must include every indexed chunk from approved sources")
    payload = {
        "version": 1, "index_config": store.index_config.metadata(),
        "records": sorted(records, key=lambda record: record['id']),
    }
    encoded = gzip.compress(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                       allow_nan=False).encode('utf-8'), mtime=0)
    # Validate a complete index before publishing the snapshot.
    with tempfile.TemporaryDirectory(prefix="corpus-validate-") as directory:
        restore_snapshot(encoded, Path(directory))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    temporary.replace(path)


def restore_snapshot(encoded: bytes, destination: Path) -> ChromaStore:
    payload = json.loads(gzip.decompress(encoded))
    if payload.get('version') != 1 or payload.get('index_config') != IndexConfig().metadata():
        raise StoreError("Corpus snapshot is incompatible with the configured embedding model")
    records = payload['records']
    if not records or len({record['id'] for record in records}) != len(records):
        raise StoreError("Corpus snapshot is empty or contains duplicate chunk IDs")
    groups = defaultdict(list)
    for record in records:
        url = record['metadata']['source_url']
        if url not in APPROVED_SOURCE_ROLES:
            raise StoreError("Corpus snapshot includes an unapproved source")
        groups[url].append(record)
    if any(scheme.seed_url not in groups for scheme in SUPPORTED_SCHEMES):
        raise StoreError("Corpus snapshot is missing a required scheme")
    store = ChromaStore(destination)
    if store.count():
        raise StoreError("Snapshot restoration requires an empty destination")
    for url, items in groups.items():
        dates = {item['metadata']['ingest_date'] for item in items}
        if len(dates) != 1:
            raise StoreError("Source snapshot has inconsistent ingestion dates")
        chunks = []
        for item in items:
            metadata = item['metadata']
            chunks.append(Chunk(
                item['id'], item['text'], metadata['scheme_name'], url,
                metadata['page_title'], metadata['chunk_index'], metadata.get('heading'),
            ))
        store.replace_source(chunks, [item['embedding'] for item in items],
                             date.fromisoformat(dates.pop()))
    if store.count() != len(records):
        raise StoreError("Corpus restoration lost chunks")
    # Exercise each source's actual vector index, not just SQLite counts.
    for url, items in groups.items():
        if not store.query(items[0]['embedding'], k=1, where={'source_url': url}):
            raise StoreError("Restored corpus failed its vector query check")
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('restore', 'export'))
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT_PATH)
    args = parser.parse_args()
    if args.command == 'restore':
        store = restore_snapshot(args.snapshot.read_bytes(), args.database)
        print(f'Restored and verified {store.count()} chunks')
    else:
        export_snapshot(ChromaStore(args.database), args.snapshot)
        print('Exported and verified portable corpus snapshot')


if __name__ == '__main__':
    main()
