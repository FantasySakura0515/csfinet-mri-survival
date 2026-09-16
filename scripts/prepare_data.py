"""Rebuild clinical metadata locally and verify the published ID-only split."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from csfinet_repro.download import download_sources
from csfinet_repro.files import read_csv, sha256
from csfinet_repro.splits import create_split


def verify(patient_csv, root=ROOT):
    identity = json.loads((root / 'data/splits/identity.json').read_text(encoding='utf-8'))
    split_path = root / 'data/splits/patients.csv'
    if sha256(split_path) != identity['id_split_sha256']:
        raise ValueError('Published ID split was modified')
    expected = read_csv(split_path, ['patient_id', 'split', 'development_split'])
    actual = read_csv(patient_csv, ['patient_id', 'split', 'development_split'])
    if [{key: row[key] for key in expected[0]} for row in actual] != expected:
        raise ValueError('Generated patient partition differs from the paper partition')
    if sha256(patient_csv) != identity['clinical_patient_csv_sha256']:
        raise ValueError('Clinical metadata differs from the pinned paper source; do not use the released models with this manifest')
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'data/raw/source-check')
    p.add_argument('--download-source', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    config = ROOT / 'configs/source-files.json'
    patients = ROOT / 'data/manifests/cohort/patients.csv'
    if args.download_source and not args.verify_only:
        download_sources(config, args.source)
    if not args.verify_only:
        sources = json.loads(config.read_text(encoding='utf-8'))
        for item in sources['files']:
            name = Path(item['path']).name
            if name in ('survival_info.csv', 'name_mapping.csv'):
                if sha256(args.source / name) != item['sha256']:
                    raise ValueError(f'Source hash mismatch: {name}')
        if not patients.exists():
            create_split(args.source, ROOT / 'configs/reconstruction-v21.json', patients.parent)
    actual = verify(patients)
    print(f'Verified {len(actual)} patients: 188 training, 47 test; internal split 150/38.')
    print('Clinical values were reconstructed locally and are excluded from Git.')


if __name__ == '__main__':
    main()
