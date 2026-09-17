"""Regression checks for the manuscript/public CSV bootstrap discrepancy."""
import itertools
import json
from pathlib import Path

import pytest

from csfinet_repro import report_v2
from csfinet_repro.files import write_csv, sha256
from csfinet_repro.segmentation_evaluation import summarize_segmentation


@pytest.mark.parametrize('variant,offset', [('raw', 0), ('tta', 18)])
def test_publication_intervals_match_manuscript_report_despite_iou_and_row_order(tmp_path, variant, offset):
    paths=[]; models={}
    for model,shift in [('csfinet',.02),('unet',0)]:
        rows=[]
        for patient in range(47):
            for region in ('WT','TC','ET'):
                value=.2+.7*(patient/47)**2+shift
                rows.append(dict(run_id='synthetic',patient_id=f'SYNTHETIC_{patient:03}',model=model,region=region,tp=1,fp=1,fn=1,dice=value,precision=value*.9,recall=value*.95,iou=value/(2-value)))
        path=tmp_path/f'{model}.csv'
        write_csv(path,list(reversed(rows)),list(rows[0]))
        paths.append(path)
        name,values=report_v2.load_segmentation_metrics(path);models[name]=values
    expected=report_v2._segmentation_rows([dict(name='synthetic',role='synthetic',models=models)],itertools.count(20260914+offset),100)
    actual=summarize_segmentation(list(reversed(paths)),tmp_path/'summary',n_resamples=100,publication_variant=variant)
    for row in expected:
        for metric in ('dice','precision','recall'):
            ci=actual['models'][row['model']][row['region']][metric]['mean_ci']
            assert ci['low']==row[f'{metric}_ci_low']
            assert ci['high']==row[f'{metric}_ci_high']


def test_published_statistics_manifest_matches_aggregate_files():
    root=Path(__file__).resolve().parents[1]/'results/reference'
    manifest=json.loads((root/'segmentation-statistics.json').read_text())
    assert manifest['n_resamples']==10000
    for name,checksum in manifest['outputs'].items():
        assert sha256(root/name)==checksum
    assert len(manifest['interval_seeds'])==48
