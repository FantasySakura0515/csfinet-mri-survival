"""Build a portable source/report bundle while indexing excluded large artifacts."""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

from .files import read_csv, sha256, write_csv, write_json


LARGE_ROLES = {
    "model_checkpoint",
    "segmentation_prediction",
    "survival_mask_cache",
    "shap_attribution",
}

RELEASE_ARTIFACTS = {
    "csfinet-repro-source-{tag}.zip",
    "csfinet-repro-reports-{tag}.zip",
    "external-artifacts.csv",
    "delivery-audit.json",
    "delivery-manifest.csv",
    "manuscript-replacements.md",
    "manuscript-claim-audit.csv",
}


def _publish_directory(staging, output, attempts=20, delay_seconds=0.25, replace=os.replace):
    """Publish a validated directory, with a Windows-safe manifest-last fallback."""
    for attempt in range(attempts):
        try:
            replace(staging, output)
            return
        except PermissionError:
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    # Some Windows filesystems reject directory replacement even when file
    # replacement works. Publish the already validated flat bundle one file at
    # a time and move the completion manifest last, so an interrupted copy is
    # never accepted by ``validate_release``.
    output.mkdir()
    try:
        children = list(staging.iterdir())
        if any(path.is_dir() for path in children):
            raise ValueError("Release staging directory must be flat")
        manifest = staging / "release-manifest.json"
        if manifest not in children:
            raise ValueError("Release staging directory has no completion manifest")
        for source in sorted((path for path in children if path != manifest), key=lambda path: path.name):
            os.replace(source, output / source.name)
        os.replace(manifest, output / manifest.name)
        staging.rmdir()
    except BaseException:
        if output.exists():
            shutil.rmtree(output)
        raise


def _git(root, *args):
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def _validate_inventory(root, rows):
    root = Path(root).resolve()
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Release inventory file missing or outside project: {row['path']}")
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"Release inventory hash/size mismatch: {row['path']}")


def _raw_nifti_index(root, split_sha):
    manifest_path = root / "data" / "manifests" / "cohort" / "nifti-downloads.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    if (not manifest.get("complete") or manifest.get("split_sha256") != split_sha
            or len(files) != 1175):
        raise ValueError("Raw NIfTI download manifest is incomplete or has the wrong split")
    rows = []
    for item in files:
        patient, modality = item["patient_id"], item["modality"]
        rows.append({
            "path": f"data/raw/brats2020-nifti/{patient}/{patient}_{modality}.nii.gz",
            "role": "raw_nifti",
            "bytes": int(item["gzip_bytes"]),
            "sha256": item["gzip_sha256"],
            "source_manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        })
    return rows


def validate_release(output, tag, split_sha):
    """Validate a completed release directory before accepting it on resume."""
    output = Path(output).resolve()
    manifest_path = output / "release-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Release manifest is missing or invalid: {error}") from error
    if (manifest.get("status") != "completed" or manifest.get("tag") != tag
            or manifest.get("split_sha256") != split_sha):
        raise ValueError("Release manifest status, tag, or split lineage does not match")
    expected = {name.format(tag=tag) for name in RELEASE_ARTIFACTS}
    artifacts = manifest.get("artifacts")
    optional = {"README_FIRST.md"}
    if (not isinstance(artifacts, dict) or not expected.issubset(artifacts)
            or set(artifacts) - expected - optional):
        raise ValueError("Release artifact set is incomplete or unexpected")
    for name, digest in artifacts.items():
        path = (output / name).resolve()
        if path.parent != output or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Release artifact is missing or has the wrong hash: {name}")
    counts = manifest.get("counts", {})
    external = read_csv(output / "external-artifacts.csv",
                        ["path", "role", "bytes", "sha256", "source_manifest"])
    if (len(external) != counts.get("external_artifacts")
            or sum(row["role"] == "raw_nifti" for row in external) != counts.get("raw_nifti_files")):
        raise ValueError("Release external-artifact counts do not match its manifest")
    archives = {
        f"csfinet-repro-source-{tag}.zip": counts.get("source_files"),
        f"csfinet-repro-reports-{tag}.zip": counts.get("report_files"),
    }
    for name, expected_files in archives.items():
        try:
            with zipfile.ZipFile(output / name) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Release ZIP failed CRC validation: {name}")
                actual_files = sum(not item.is_dir() for item in archive.infolist())
                if manifest.get("source_pdf_policy") == "excluded" and any(
                        item.filename.lower().endswith(".pdf") for item in archive.infolist()):
                    raise ValueError(f"Release unexpectedly contains a PDF: {name}")
        except zipfile.BadZipFile as error:
            raise ValueError(f"Release ZIP is invalid: {name}") from error
        if actual_files != expected_files:
            raise ValueError(f"Release ZIP file count does not match its manifest: {name}")
    return manifest


def create_release(project_root=".", tag="v2", output=None):
    if not tag.isalnum():
        raise ValueError("Tag must be alphanumeric")
    root = Path(project_root).resolve()
    output = Path(output).resolve() if output else root / "release" / tag
    if output.exists():
        raise FileExistsError("Release output exists; release bundles are immutable")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked files are modified; commit them before creating a release")
    commit = _git(root, "rev-parse", "HEAD")

    delivery_dir = root / "results" / "delivery" / tag
    audit_path = delivery_dir / "delivery-audit.json"
    inventory_path = delivery_dir / "delivery-manifest.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "completed" or audit.get("tag") != tag:
        raise ValueError("Delivery audit is not completed for this tag")
    if sha256(inventory_path) != audit.get("manifest_sha256"):
        raise ValueError("Delivery manifest does not match the completed audit")
    inventory = read_csv(inventory_path, ["path", "role", "bytes", "sha256"])
    _validate_inventory(root, inventory)

    split_sha = audit["split_sha256"]
    large = [dict(row, source_manifest=str(inventory_path.relative_to(root)).replace("\\", "/"))
             for row in inventory if row["role"] in LARGE_ROLES]
    large.extend(_raw_nifti_index(root, split_sha))
    large.sort(key=lambda row: (row["role"], row["path"]))
    reports = [row for row in inventory if row["role"] not in LARGE_ROLES]

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)).resolve()
    try:
        source_archive = staging / f"csfinet-repro-source-{tag}.zip"
        subprocess.run(
            ["git", "archive", "--format=zip", f"--output={source_archive}", commit],
            cwd=root, check=True,
        )
        reports_archive = staging / f"csfinet-repro-reports-{tag}.zip"
        with zipfile.ZipFile(reports_archive, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for row in sorted(reports, key=lambda value: value["path"]):
                archive.write(root / row["path"], arcname=row["path"])

        external_path = staging / "external-artifacts.csv"
        write_csv(external_path, large, ["path", "role", "bytes", "sha256", "source_manifest"])
        copied = (audit_path, inventory_path,
                  delivery_dir / "manuscript-replacements.md",
                  delivery_dir / "manuscript-claim-audit.csv")
        if (delivery_dir / "README_FIRST.md").exists():
            copied += (delivery_dir / "README_FIRST.md",)
        for source in copied:
            shutil.copy2(source, staging / source.name)

        with zipfile.ZipFile(source_archive) as archive:
            source_files = sum(not item.is_dir() for item in archive.infolist())
            if any(item.filename.lower().endswith(".pdf") for item in archive.infolist()):
                raise ValueError("Source archive contains a PDF; set export-ignore before releasing")
        result = {
            "status": "completed",
            "scope": "portable_source_reports_and_external_artifact_index",
            "tag": tag,
            "git_commit": commit,
            "split_sha256": split_sha,
            "counts": {
                "source_files": source_files,
                "report_files": len(reports),
                "external_artifacts": len(large),
                "raw_nifti_files": 1175,
            },
            "artifacts": {
                source_archive.name: sha256(source_archive),
                reports_archive.name: sha256(reports_archive),
                external_path.name: sha256(external_path),
                **{source.name: sha256(staging / source.name) for source in copied},
            },
            "author_review_still_required": bool(audit.get("author_review_still_required", True)),
            "source_pdf_policy": "excluded",
            "report_revision_of": audit.get("revision_of"),
        }
        write_json(staging / "release-manifest.json", result)
        validate_release(staging, tag, split_sha)
        _publish_directory(staging, output)
        validate_release(output, tag, split_sha)
        return result
    except BaseException:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise
