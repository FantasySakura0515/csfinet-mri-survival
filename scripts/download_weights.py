"""Download and verify the paper's six checkpoints from this repository's Release."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def archive_files(archive):
    files = {m[key]: m[sha] for m in archive['models']
             for key, sha in [('checkpoint_path', 'checkpoint_sha256'), ('run_path', 'run_sha256')]}
    for item in archive.get('license_files', []):
        if item['path'] not in ('LICENSE', 'NOTICE') or item['path'] in files:
            raise ValueError('Unexpected or duplicated license path')
        files[item['path']] = item['sha256']
    return files


def install_archive(path, archive, root=ROOT):
    if digest(path) != archive['sha256']:
        raise ValueError(f'Archive checksum mismatch: {path}')
    expected = archive_files(archive)
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(expected) or set(z.namelist()) != set(expected):
            raise ValueError('Archive contains unexpected or duplicated files')
        for entry in z.infolist():
            rel = PurePosixPath(entry.filename)
            target = (root / rel).resolve()
            if (rel.is_absolute() or '..' in rel.parts or not target.is_relative_to(root.resolve())
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError('Unsafe archive member')
            if target.exists():
                if digest(target) != expected[entry.filename]:
                    raise FileExistsError(f'Will not overwrite a different local file: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_suffix(target.suffix + '.part')
            try:
                with z.open(entry) as source, temp.open('wb') as out:
                    shutil.copyfileobj(source, out)
                if digest(temp) != expected[entry.filename]:
                    raise ValueError(f'Extracted file checksum mismatch: {entry.filename}')
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)


def verify_archive(archive, root=ROOT):
    for name, expected in archive_files(archive).items():
        if digest(root / name) != expected:
            raise ValueError(f'Installed file checksum mismatch: {name}')
    for model in archive['models']:
        if digest(root / model['config_path']) != model['config_sha256']:
            raise ValueError(f'Configuration was changed: {model["config_path"]}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--component', choices=['all', 'segmentation', 'survival'], default='all')
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--archive-dir', type=Path, help='Install already downloaded ZIPs instead of fetching them')
    args = p.parse_args()
    manifest = json.loads((ROOT / 'release-manifest.json').read_text(encoding='utf-8'))
    for archive in manifest['archives']:
        if args.component != 'all' and not archive['filename'].startswith(args.component + '-'):
            continue
        if not args.verify_only:
            directory = args.archive_dir or ROOT / 'downloads'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / archive['filename']
            if not path.exists():
                if args.archive_dir:
                    raise FileNotFoundError(path)
                url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}/{archive['filename']}"
                print(f'Downloading {archive["filename"]} ({archive["size_bytes"] / 1048576:.1f} MiB)', flush=True)
                temp = path.with_suffix('.zip.part')
                try:
                    with urlopen(Request(url, headers={'User-Agent': 'csfinet-mri-survival/2.0'}), timeout=120) as source, temp.open('wb') as out:
                        shutil.copyfileobj(source, out)
                    if digest(temp) != archive['sha256']:
                        raise ValueError('Downloaded archive checksum mismatch')
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
            install_archive(path, archive)
        verify_archive(archive)
        print(f'Verified {archive["filename"]}: {len(archive["models"])} models')


if __name__ == '__main__':
    main()
