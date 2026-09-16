"""Evaluate released fixed models; never train or select models on the test set."""
import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from download_weights import verify_archive
from prepare_data import verify


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['segmentation', 'survival', 'shap', 'all'], default='segmentation')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--raw-root', type=Path, default=ROOT / 'data/raw/brats2020-nifti')
    p.add_argument('--limit', type=int, help='Smoke check only: 1..46 test patients, isolated outputs under */smoke/')
    p.add_argument('--slice-batch', type=int, help='Optional smaller inference batch for limited GPU memory')
    p.add_argument('--resume', action='store_true', help='Resume only outputs matching the same inputs and weights')
    args = p.parse_args()
    if args.limit is not None and not 1 <= args.limit < 47:
        p.error('--limit must be 1..46; omit it for the formal 47-patient evaluation')
    if args.limit and args.stage in ('shap', 'all'):
        p.error('SHAP requires all 47 patients; smoke checks support segmentation or survival stages')
    if args.slice_batch is not None and args.slice_batch < 1:
        p.error('--slice-batch must be positive')

    import numpy as np
    import torch
    from csfinet_repro.files import read_csv, sha256, write_csv, write_json
    from csfinet_repro.training import seed_everything
    patients_path = ROOT / 'data/manifests/cohort/patients.csv'
    patients = verify(patients_path)
    tests = [r for r in patients if r['split'] == 'test'][:args.limit]
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; install an appropriate PyTorch build or use --device cpu')
    seed_everything(20260914)
    torch.set_num_threads(4)
    results = ROOT / ('results/smoke' if args.limit else 'results')
    artifacts = ROOT / ('artifacts/smoke' if args.limit else 'artifacts')
    release = json.loads((ROOT / 'release-manifest.json').read_text(encoding='utf-8'))
    components = ('segmentation', 'survival') if args.stage == 'all' else (('survival',) if args.stage == 'survival' else ('segmentation',))
    for archive in release['archives']:
        if any(archive['filename'].startswith(name + '-') for name in components):
            verify_archive(archive)
    models_by_name = {m['model']: m for a in release['archives'] for m in a['models']}

    def js(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def start(directory, identity):
        target = directory / 'release-inputs.json'
        if directory.exists():
            if not args.resume or not target.exists() or js(target) != identity:
                raise FileExistsError(f'Outputs already exist without matching --resume: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        write_json(target, identity)

    from csfinet_repro.nifti import load_patient, MODALITIES
    input_hashes = {r['patient_id']: {m: sha256(args.raw_root / r['patient_id'] / f'{r["patient_id"]}_{m}.nii.gz')
                                     for m in (*MODALITIES, 'seg')} for r in tests}
    common = dict(split_sha256=sha256(patients_path), test_ids=[r['patient_id'] for r in tests],
                  inputs=input_hashes, device=str(device), torch_version=torch.__version__,
                  scope='smoke_only' if args.limit else 'released_fixed_models_47_patient_evaluation')

    if args.stage in ('segmentation', 'all'):
        from csfinet_repro.models import build_segmentation
        from csfinet_repro.segmentation import original_labels, predict_patient
        from csfinet_repro.inference_variants import predict_labels, FLIPS_TTA
        from csfinet_repro.metrics import region_metrics
        from csfinet_repro.segmentation_evaluation import save_prediction, summarize_segmentation
        config_path = ROOT / 'configs/reconstruction-v21.json'
        config = js(config_path)
        batch = args.slice_batch or config['segmentation']['slice_batch']
        combined = []
        for name in ('csfinet', 'unet'):
            info = models_by_name[name]
            directory = results / f'segmentation/{name}-test-v2'
            identity = dict(**common, model=name, config_sha256=sha256(config_path),
                            checkpoint_sha256=info['checkpoint_sha256'], slice_batch=batch)
            start(directory, identity)
            run = js(ROOT / info['run_path'])
            if run['split_sha256'] != common['split_sha256']:
                raise ValueError('Released model split mismatch')
            model = build_segmentation(name, config).to(device).eval()
            model.load_state_dict(torch.load(ROOT / info['checkpoint_path'], map_location='cpu', weights_only=True))
            raw_rows, tta_rows, records = [], [], []
            for i, row in enumerate(tests, 1):
                pid = row['patient_id']
                images, target = load_patient(args.raw_root, pid)
                truth = original_labels(target)
                progress = directory / 'progress' / f'{pid}.json'
                cached = js(progress) if args.resume and progress.exists() else None
                if cached:
                    for item in cached['files']:
                        if sha256(artifacts / item['path']) != item['sha256']:
                            raise ValueError(f'Cached prediction modified: {pid}')
                    raw_rows.extend(cached['raw_rows']); tta_rows.extend(cached['tta_rows'])
                    records.append(cached['raw_artifact'])
                else:
                    files, rr, tr = [], [], []
                    for variant in ('raw', 'tta'):
                        pred = (original_labels(predict_patient(model, images, batch, device)) if variant == 'raw'
                                else predict_labels(model, images, batch, device, flips=FLIPS_TTA))
                        rel = f'segmentation/{name}-test-v2/{pid}/{pid}_{name}_prediction.nii.gz' if variant == 'raw' else f'segmentation/{name}-tta-v2/{pid}/{pid}_{name}_prediction.nii.gz'
                        file_hash = save_prediction(args.raw_root / pid / f'{pid}_seg.nii.gz', pred, artifacts / rel)
                        files.append(dict(path=rel, sha256=file_hash))
                        for region, measures in region_metrics(truth, pred).items():
                            entry = dict(run_id=f'{name}-refit-v2', patient_id=pid, model=name, region=region, **measures)
                            (rr if variant == 'raw' else tr).append(entry)
                        if variant == 'raw':
                            record = dict(patient_id=pid, relative_path=rel.removeprefix('segmentation/'),
                                          prediction_sha256=file_hash, truth_sha256=input_hashes[pid]['seg'])
                        del pred
                    raw_rows.extend(rr); tta_rows.extend(tr); records.append(record)
                    write_json(progress, dict(files=files, raw_rows=rr, tta_rows=tr, raw_artifact=record))
                del images, target, truth
                print(f'{name}: raw + matched TTA {i}/{len(tests)} {pid}', flush=True)
            write_csv(directory / 'segmentation_patient_metrics.csv', raw_rows, list(raw_rows[0]))
            tta_dir = results / f'inference-variants/v2/test/per-variant'
            write_csv(tta_dir / f'{name}_tta_patient_metrics.csv', tta_rows, list(tta_rows[0]))
            combined.extend(dict(r, variant=v) for v, rows in [('raw', raw_rows), ('tta', tta_rows)] for r in rows)
            write_json(directory / 'inference-manifest.json', dict(status='completed', **identity,
                       refit_run_sha256=sha256(ROOT / info['run_path']), artifacts=records,
                       test_access='Re-evaluation of published fixed weights; no model selection.'))
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        write_csv(results / 'inference-variants/v2/test/test_patient_metrics.csv', combined, list(combined[0]))
        if not args.limit:
            summarize_segmentation([results / f'segmentation/{m}-test-v2/segmentation_patient_metrics.csv' for m in ('csfinet', 'unet')], results / 'tables/v2/segmentation')
            summarize_segmentation([results / f'inference-variants/v2/test/per-variant/{m}_tta_patient_metrics.csv' for m in ('csfinet', 'unet')], results / 'inference-variants/v2/test/table5b')

    if args.stage in ('survival', 'all'):
        from csfinet_repro.models import SurvivalNet
        from csfinet_repro.survival import VARIANTS, clinical_features
        from csfinet_repro.survival_mri import build_mri_volume, load_wt
        config_path = ROOT / 'configs/survival-v2.json'
        config = js(config_path)['survival']
        mask_dir = artifacts / 'segmentation/csfinet-test-v2'
        seg_manifest_path = results / 'segmentation/csfinet-test-v2/inference-manifest.json'
        seg_manifest = js(seg_manifest_path)
        if (seg_manifest.get('status') != 'completed' or seg_manifest.get('model') != 'csfinet'
                or seg_manifest.get('checkpoint_sha256') != models_by_name['csfinet']['checkpoint_sha256']
                or seg_manifest.get('split_sha256') != common['split_sha256']
                or seg_manifest.get('test_ids') != common['test_ids']
                or seg_manifest.get('inputs') != common['inputs']):
            raise ValueError('Expected raw v2 CSFINet masks from the released checkpoint')
        mask_hashes = {r['patient_id']: r['prediction_sha256'] for r in seg_manifest['artifacts']}
        for row in tests:
            pid = row['patient_id']
            if sha256(mask_dir / pid / f'{pid}_csfinet_prediction.nii.gz') != mask_hashes[pid]:
                raise ValueError(f'Segmentation mask hash mismatch: {pid}')
        directory = results / 'survival/v2mask-fixed'
        start(directory, dict(**common, config_sha256=sha256(config_path), masks=mask_hashes,
                              weights={v: models_by_name[v]['checkpoint_sha256'] for v in VARIANTS}))
        models, fitted = {}, {}
        train = [r for r in patients if r['split'] == 'train']
        for variant, n in VARIANTS.items():
            info = models_by_name[variant]
            run = js(ROOT / info['run_path'])
            if run['split_sha256'] != common['split_sha256'] or set(run['fitted_clinical']['training_ids']) != {r['patient_id'] for r in train}:
                raise ValueError('Survival preprocessing/split mismatch')
            fitted[variant] = run['fitted_clinical']
            models[variant] = SurvivalNet(n, tuple(config['cnn_channels']), config['dropout'], in_channels=5).to(device).eval()
            models[variant].load_state_dict(torch.load(ROOT / info['checkpoint_path'], map_location='cpu', weights_only=True))
        ages = np.array([float(r['age']) for r in train])
        y = np.array([float(r['survival_days']) for r in train])
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(train)), ages]), y, rcond=None)[0]
        output = []
        with torch.inference_mode():
            for i, row in enumerate(tests, 1):
                pid = row['patient_id']
                images, _ = load_patient(args.raw_root, pid)
                volume = build_mri_volume(images, load_wt(mask_dir / pid / f'{pid}_csfinet_prediction.nii.gz'))
                x = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).to(device)
                pred = {}
                for variant, model in models.items():
                    clinical = torch.from_numpy(clinical_features([row], fitted[variant], variant)).to(device) if VARIANTS[variant] else None
                    pred[variant] = float(model(x, clinical).item())
                pred.update(training_mean=float(y.mean()), age_ols=float(coef[0] + coef[1] * float(row['age'])))
                for variant, value in pred.items():
                    truth = float(row['survival_days'])
                    output.append(dict(run_id='v2mask-fixed', patient_id=pid, model=variant, y_true=truth, y_pred=value,
                                       residual=value-truth, absolute_error=abs(value-truth), age=row['age'], resection_status=row['resection_status'],
                                       mask_source='training_only_clinical_reference' if variant in ('training_mean', 'age_ols') else 'predicted_CSFINet_WT_v2_raw'))
                del images, volume, x
                print(f'Survival {i}/{len(tests)} {pid}', flush=True)
        write_csv(directory / 'survival_predictions.csv', output, list(output[0]))
        write_json(directory / 'manifest.json', dict(status='completed', **common, retrained=False,
                   scope_note='Post-hoc input substitution on previously inspected test patients; fixed survival weights, raw v2 masks.',
                   segmentation_manifest_sha256=sha256(seg_manifest_path), predictions_sha256=sha256(directory / 'survival_predictions.csv')))
        if not args.limit:
            from csfinet_repro import report_v2 as report
            experiments = [dict(name='v2mask-fixed', role='post_hoc_fixed_model_input_substitution', groups=report.load_survival_predictions(directory / 'survival_predictions.csv'))]
            seeds = itertools.count(20260916)
            summary = report._survival_rows(experiments, seeds, 10000)
            write_csv(directory / 'survival_comparison.csv', summary, list(summary[0]))
            degeneracy = report._degeneracy_rows(experiments)
            write_csv(directory / 'survival_degeneracy.csv', degeneracy, list(degeneracy[0]))
            paired = report._survival_comparisons(experiments, seeds, 10000)
            for row in paired:
                row['role'] = 'post_hoc_reference_comparison'
            write_csv(directory / 'survival_paired_tests.csv', paired, list(paired[0]))

    if args.stage in ('shap', 'all'):
        from csfinet_repro.explainability import explain_test, parameter_randomization, summarize_table9
        config = ROOT / 'configs/reconstruction-v21.json'
        analysis = ROOT / 'configs/analysis.json'
        run = ROOT / 'runs/csfinet-refit-v2'
        out = results / 'explainability/v2'
        explain_test(config, analysis, patients_path, args.raw_root, run,
                     artifacts / 'segmentation/csfinet-test-v2', results / 'segmentation/csfinet-test-v2/inference-manifest.json',
                     out, artifacts / 'explainability/v2', resume=args.resume, device_override=str(device))
        parameter_randomization(config, analysis, patients_path, args.raw_root, run,
                                out / 'parameter_randomization.csv', resume=args.resume, device_override=str(device))
        summarize_table9(out / 'shap_patient_metrics.csv', results / 'segmentation/csfinet-test-v2/segmentation_patient_metrics.csv',
                         out / 'parameter_randomization.csv', results / 'tables/v2/shap')
    print('Completed. Smoke outputs are not paper results.' if args.limit else 'Completed released-model evaluation.')


if __name__ == '__main__':
    main()
