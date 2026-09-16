import csv
import json
from pathlib import Path
import subprocess
import zipfile

from csfinet_repro.files import sha256
import pytest

from csfinet_repro.release import _publish_directory, create_release, validate_release


def test_publish_directory_retries_transient_windows_lock(tmp_path):
    staging, output = tmp_path / "staging", tmp_path / "published"
    staging.mkdir()
    (staging / "manifest.json").write_text("{}", encoding="utf-8")
    calls = []

    def transient_replace(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            raise PermissionError("temporary scanner lock")
        source.rename(destination)

    _publish_directory(staging, output, attempts=3, delay_seconds=0, replace=transient_replace)
    assert len(calls) == 3
    assert (output / "manifest.json").is_file()


def test_publish_directory_uses_manifest_last_fallback(tmp_path):
    staging, output = tmp_path / "staging", tmp_path / "published"
    staging.mkdir()
    (staging / "report.zip").write_bytes(b"report")
    (staging / "release-manifest.json").write_text("{}", encoding="utf-8")

    def directory_replace_unavailable(_source, _destination):
        raise PermissionError("directory replacement unavailable")

    _publish_directory(staging, output, attempts=1, delay_seconds=0,
                       replace=directory_replace_unavailable)
    assert not staging.exists()
    assert (output / "report.zip").read_bytes() == b"report"
    assert (output / "release-manifest.json").is_file()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_release_separates_reports_from_large_artifacts(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    tracked = root / "README.md"
    tracked.write_text("repro\n", encoding="utf-8")
    (root / "private.pdf").write_bytes(b"private manuscript")
    (root / ".gitattributes").write_text("*.pdf export-ignore\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md", "private.pdf", ".gitattributes"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "test"], cwd=root, check=True)

    report = root / "results" / "tables" / "v2" / "table.csv"
    report.parent.mkdir(parents=True)
    report.write_text("metric,value\nMAE,1\n", encoding="utf-8")
    checkpoint = root / "runs" / "model" / "final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"weights")
    delivery = root / "results" / "delivery" / "v2"
    rows = [
        {"path": str(report.relative_to(root)).replace("\\", "/"), "role": "manuscript_table",
         "bytes": report.stat().st_size, "sha256": sha256(report)},
        {"path": str(checkpoint.relative_to(root)).replace("\\", "/"), "role": "model_checkpoint",
         "bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)},
    ]
    inventory = delivery / "delivery-manifest.csv"
    _write_csv(inventory, rows)
    split_sha = "a" * 64
    (delivery / "manuscript-replacements.md").write_text("replacement\n", encoding="utf-8")
    _write_csv(delivery / "manuscript-claim-audit.csv", [{"section": "Results"}])
    (delivery / "delivery-audit.json").write_text(json.dumps({
        "status": "completed", "tag": "v2", "split_sha256": split_sha,
        "manifest_sha256": sha256(inventory), "author_review_still_required": True,
    }), encoding="utf-8")
    raw_manifest = root / "data" / "manifests" / "cohort" / "nifti-downloads.json"
    raw_manifest.parent.mkdir(parents=True)
    raw_manifest.write_text(json.dumps({
        "complete": True, "split_sha256": split_sha,
        "files": [{"patient_id": f"p{i:03d}", "modality": "t1", "gzip_bytes": i + 1,
                   "gzip_sha256": f"{i:064x}"} for i in range(1175)],
    }), encoding="utf-8")

    output = root / "bundle"
    result = create_release(root, "v2", output)
    assert result["counts"]["report_files"] == 1
    assert result["counts"]["source_files"] == 2
    with zipfile.ZipFile(output / "csfinet-repro-source-v2.zip") as archive:
        assert not any(name.endswith(".pdf") for name in archive.namelist())
    assert result["counts"]["external_artifacts"] == 1176
    with zipfile.ZipFile(output / "csfinet-repro-reports-v2.zip") as archive:
        assert archive.namelist() == ["results/tables/v2/table.csv"]
    with (output / "external-artifacts.csv").open(encoding="utf-8", newline="") as stream:
        external = list(csv.DictReader(stream))
    assert {row["role"] for row in external} == {"model_checkpoint", "raw_nifti"}
    manifest = json.loads((output / "release-manifest.json").read_text(encoding="utf-8"))
    assert manifest["git_commit"]
    assert set(manifest["artifacts"]) == {
        "csfinet-repro-source-v2.zip", "csfinet-repro-reports-v2.zip", "external-artifacts.csv",
        "delivery-audit.json", "delivery-manifest.csv", "manuscript-replacements.md",
        "manuscript-claim-audit.csv",
    }
    assert validate_release(output, "v2", split_sha) == manifest

    (output / "manuscript-replacements.md").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong hash"):
        validate_release(output, "v2", split_sha)
