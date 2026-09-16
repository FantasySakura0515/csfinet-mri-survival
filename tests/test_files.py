import csv
import json

import pytest

from csfinet_repro.files import write_csv, write_json


def test_atomic_json_preserves_existing_file_when_serialization_fails(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"status": "good"})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert path.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "good"}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_csv_preserves_existing_file_when_a_row_is_invalid(tmp_path):
    path = tmp_path / "table.csv"
    write_csv(path, [{"a": 1}], ["a"])
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_csv(path, [{"a": 2, "unexpected": 3}], ["a"])
    assert path.read_bytes() == original
    with path.open(encoding="utf-8", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"a": "1"}]
    assert list(tmp_path.glob(".*.tmp")) == []
