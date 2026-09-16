"""Version-pinned small source downloads; does not fetch the full MRI dataset."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from .files import sha256, write_json


def unpack_download(payload, expected_name, limit):
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            matches = [entry for entry in archive.infolist()
                       if not entry.is_dir() and PurePosixPath(entry.filename).name == expected_name]
            if len(matches) != 1 or matches[0].file_size > limit:
                raise ValueError("Unexpected or oversized archive member")
            payload = archive.read(matches[0])
    if len(payload) > limit or payload.lstrip().startswith(b"<"):
        raise ValueError("Expected a small data file; received HTML or oversized content")
    return payload


def download_sources(config_path, destination):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["dataset"] != "awsaf49/brats2020-training-data":
        raise ValueError("This downloader is scoped to the user-designated dataset")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    names = [PurePosixPath(item["path"]).name for item in config["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate local filenames in source configuration")
    for item in config["files"]:
        remote = item["path"]
        target = destination / PurePosixPath(remote).name
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise ValueError(f"Existing source has unexpected hash: {target}; keep it for investigation")
        else:
            url = (f"https://www.kaggle.com/api/v1/datasets/download/{config['dataset']}/"
                   f"{quote(remote, safe='')}?datasetVersionNumber={int(config['version'])}")
            with urlopen(Request(url, headers={"User-Agent": "CSFINet-reproduction-source-audit"}), timeout=30) as response:
                payload = response.read(config["max_file_bytes"] + 1)
            if len(payload) > config["max_file_bytes"]:
                raise ValueError(f"Download exceeded size cap: {remote}")
            payload = unpack_download(payload, target.name, config["max_file_bytes"])
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded content hash mismatch: {remote}")
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
        records.append(dict(remote_path=remote, local_name=target.name, bytes=target.stat().st_size,
                            sha256=sha256(target)))
    manifest = dict(dataset=config["dataset"], version=config["version"],
                    config_sha256=sha256(config_path), files=records)
    write_json(destination / "source-downloads.json", manifest)
    return manifest
