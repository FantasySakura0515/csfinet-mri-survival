"""Rebuild manuscript segmentation summaries from saved patient metrics, without inference."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def summarize(results, output):
    from csfinet_repro import analysis_report as report
    from csfinet_repro.files import sha256, write_csv, write_json
    from csfinet_repro.segmentation_evaluation import summarize_segmentation
    import numpy, scipy
    results, output = Path(results), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT/'data/splits/patients.csv').open(encoding='utf-8-sig',newline='') as f:
        expected_ids={r['patient_id'] for r in csv.DictReader(f) if r['split']=='test'}
    experiments, sources, seeds = [], [], []
    for variant in ('raw', 'tta'):
        paths = [results / (f'segmentation/{m}-test-study/segmentation_patient_metrics.csv' if variant == 'raw' else f'inference-variants/study/test/per-variant/{m}_tta_patient_metrics.csv') for m in ('csfinet', 'unet')]
        models = dict(report.load_segmentation_metrics(p) for p in paths)
        if set(models)!=set(('csfinet','unet')) or any({pid for pid,region in values}!=expected_ids for values in models.values()):
            raise ValueError('Summary inputs must contain the published 47-patient test partition for both models')
        directory = output / variant
        summarize_segmentation(paths, directory, publication_variant=variant)
        shutil.copy2(directory/'table5.csv', output/f'segmentation-{variant}.csv')
        experiments.append(dict(name='study' if variant=='raw' else 'study_tta', tag='study', variant=None if variant=='raw' else 'tta', role='raw_inference' if variant=='raw' else 'validation_selected_tta', models=models, complete=True))
        sources.extend(dict(path=p.relative_to(results).as_posix(), sha256=sha256(p)) for p in paths)
        detail = json.loads((directory/'table5.json').read_text(encoding='utf-8'))
        for model in ('csfinet','unet'):
            for region in ('WT','TC','ET'):
                for measure in ('dice','precision','recall','iou'):
                    ci=detail['models'][model][region][measure]['mean_ci']
                    seeds.append(dict(variant=variant,model=model,region=region,measure=measure,seed=ci['seed']))
    # This fixed offset preserves the manuscript's paired-analysis schedule;
    # adding optional diagnostic variants cannot silently shift its seeds.
    paired=report._segmentation_comparisons(experiments,itertools.count(20260964),10000)
    write_csv(output/'segmentation-paired-tests.csv',paired,list(paired[0]))
    write_json(output/'segmentation-statistics.json',dict(method='patient-level mean; sample SD ddof=1; pointwise 95% BCa',n_resamples=10000,base_seed=20260914,paired_base_seed=20260964,patient_order='lexicographic patient_id',numpy=numpy.__version__,scipy=scipy.__version__,sources=sources,interval_seeds=seeds,outputs={p.name:sha256(p) for p in output.glob('*.csv')}))
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'results/publication')
    args=parser.parse_args()
    print(summarize(args.results,args.output))
