from datetime import date
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.config import SUPPORTED_SCHEMES
from src.corpus_snapshot import export_snapshot, restore_snapshot
from src.models import Chunk
from src.store import ChromaStore, StoreError


def make_store(path: Path) -> ChromaStore:
    store = ChromaStore(path)
    for index, scheme in enumerate(SUPPORTED_SCHEMES):
        chunk = Chunk(f'chunk-{index}', 'Expense ratio: 1.03%', scheme.canonical_name,
                      scheme.seed_url, 'Fund page', 0)
        vector = [0.0] * 384
        vector[index] = 1.0
        store.replace_source([chunk], [vector], date(2026, 10, 2))
    return store


def test_snapshot_restores_vectors_in_a_separate_process(tmp_path):
    store = make_store(tmp_path / 'original')
    snapshot = tmp_path / 'corpus.json.gz'
    export_snapshot(store, snapshot)
    code = '''
import sys
from pathlib import Path
from src.corpus_snapshot import restore_snapshot
store = restore_snapshot(Path(sys.argv[1]).read_bytes(), Path(sys.argv[2]))
hits = store.query([1.0] + [0.0] * 383, k=1)
assert store.count() == 5
assert hits[0].chunk_id == 'chunk-0'
assert hits[0].text == 'Expense ratio: 1.03%'
assert hits[0].ingest_date.isoformat() == '2026-10-02'
'''
    subprocess.run([sys.executable, '-c', code, str(snapshot), str(tmp_path / 'fresh')],
                   check=True, capture_output=True, text=True)


def test_new_snapshot_has_no_deleted_vectors_and_does_not_modify_old_index(tmp_path):
    store = make_store(tmp_path / 'original')
    snapshot = tmp_path / 'corpus.json.gz'
    export_snapshot(store, snapshot)
    old = restore_snapshot(snapshot.read_bytes(), tmp_path / 'old')
    scheme = SUPPORTED_SCHEMES[0]
    chunk = Chunk('new-id', 'Expense ratio: 0.5%', scheme.canonical_name, scheme.seed_url,
                  'Fund page', 0)
    store.replace_source([chunk], [[1.0] + [0.0] * 383], date(2026, 10, 3))
    export_snapshot(store, snapshot)
    new = restore_snapshot(snapshot.read_bytes(), tmp_path / 'new')
    assert new.query([1.0] + [0.0] * 383, k=1)[0].chunk_id == 'new-id'
    assert old.query([1.0] + [0.0] * 383, k=1)[0].chunk_id == 'chunk-0'
    assert new.count() == old.count() == 5


def test_incompatible_snapshot_is_rejected_before_opening_store(tmp_path):
    snapshot = tmp_path / 'corpus.json.gz'
    export_snapshot(make_store(tmp_path / 'original'), snapshot)
    payload = json.loads(gzip.decompress(snapshot.read_bytes()))
    payload['index_config']['embedding_revision'] = 'wrong-model'
    with pytest.raises(StoreError, match='incompatible'):
        restore_snapshot(gzip.compress(json.dumps(payload).encode()), tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
