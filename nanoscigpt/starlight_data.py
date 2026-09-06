"""Prepare the public StarLightCurves archive and run its classroom baseline."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

URL = 'https://www.timeseriesclassification.com/aeon-toolkit/StarLightCurves.zip'
ARCHIVE_SHA256 = 'f5a3b1646837eabee4b292313263fc40a00524df9ab14e0c4c901829f63bbf33'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_split(payload):
    rows = np.loadtxt(io.BytesIO(payload), dtype=np.float64, ndmin=2)
    if rows.shape[1] != 1025 or not np.isfinite(rows).all():
        raise ValueError('Expected one label and 1024 finite brightness values per row')
    if not np.isin(rows[:, 0], [1, 2, 3]).all():
        raise ValueError('Expected original integer labels 1, 2, 3')
    return rows[:, 1:].astype('float32'), rows[:, 0].astype('int64')


def prepare(archive, output):
    archive, out = Path(archive), Path(output)
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Archive SHA256 differs from the verified classroom release')
    with zipfile.ZipFile(archive) as source:
        train_bytes = source.read('StarLightCurves_TRAIN.txt')
        test_bytes = source.read('StarLightCurves_TEST.txt')
        description = source.read('StarLightCurves.txt').decode('utf-8')
    x, y = parse_split(train_bytes)
    xt, yt = parse_split(test_bytes)
    if x.shape != (1000, 1024) or xt.shape != (8236, 1024):
        raise ValueError('Unexpected original benchmark sample counts')
    fit, val = train_test_split(np.arange(len(y)), test_size=.2, random_state=42, stratify=y)
    train_only = dict(x_train=x, y_train=y, fit_indices=fit, validation_indices=val)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'data-and-splits.npz', **train_only, x_test=xt, y_test=yt)
    np.savez_compressed(out / 'train-validation-only.npz', **train_only)
    (out / 'source-description.txt').write_text(description, encoding='utf-8')
    counts = lambda a: {str(c): int((a == c).sum()) for c in (1, 2, 3)}
    manifest = dict(source_url=URL, archive_sha256=ARCHIVE_SHA256,
        train_file_sha256=hashlib.sha256(train_bytes).hexdigest(),
        test_file_sha256=hashlib.sha256(test_bytes).hexdigest(),
        fit_samples=len(fit), validation_samples=len(val), test_samples=len(yt),
        points_per_curve=1024, class_counts=dict(fit=counts(y[fit]), validation=counts(y[val]), test=counts(yt)),
        label_status='All original rows have labels. Ignoring labels during pretraining is a teaching setup, not a newly collected unlabeled dataset.',
        boundary='Preprocessed classification benchmark, not raw survey data. Class-name mapping, star IDs and physical time units are not established by this package.',
        files={name: sha256(out / name) for name in ['data-and-splits.npz', 'train-validation-only.npz']})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def baseline(data, output):
    # Reuse the same split validation as the scientific-language experiment.
    from .starlight import load_data
    splits = load_data(data)
    arrays = {k: (x.numpy(), y.numpy() + 1) for k, (x, y) in splits.items()}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    x, y = arrays['fit']
    predictions, results = {}, {}
    models = dict(majority=DummyClassifier(strategy='most_frequent'),
        random_forest=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2,
                                            class_weight='balanced', max_features='sqrt'))
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(x, y)
        results[name] = {}
        for part in ('val', 'test'):
            values, labels = arrays[part]
            pred = model.predict(values)
            results[name][part] = dict(accuracy=float(accuracy_score(labels, pred)),
                macro_f1=float(f1_score(labels, pred, average='macro')),
                confusion_matrix=confusion_matrix(labels, pred, labels=[1, 2, 3]).tolist(),
                per_class=classification_report(labels, pred, output_dict=True, zero_division=0))
            if part == 'test':
                predictions[name] = pred
        results[name]['seconds'] = time.perf_counter() - start
        print(json.dumps(dict(model=name, **results[name])), flush=True)
    joblib.dump(models['random_forest'], out / 'random_forest.joblib')
    np.savez_compressed(out / 'predictions.npz', y_true=arrays['test'][1], **predictions)
    report = dict(data=str(Path(data).resolve()), data_sha256=sha256(data),
        fit_samples=len(x), validation_samples=len(arrays['val'][0]), test_samples=len(arrays['test'][0]),
        class_ids=[1, 2, 3], results=results, model_parameters=models['random_forest'].get_params(),
        boundary='Direct supervised classification, no pretraining. Benchmark accuracy is not a physical-mechanism claim.')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--archive', default='out/starlight/StarLightCurves.zip')
    p.add_argument('--download', action='store_true', help='Download the public archive if absent (about 110 MB)')
    p.add_argument('--output', required=True)
    p = sub.add_parser('baseline')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'baseline':
        baseline(args.data, args.output)
    else:
        archive = Path(args.archive)
        if args.download and not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive partial file retains failed downloads without replacing old data.
            partial = archive.with_suffix('.zip.partial')
            with partial.open('xb') as target, urllib.request.urlopen(URL, timeout=60) as source:
                shutil.copyfileobj(source, target)
            if sha256(partial) != ARCHIVE_SHA256:
                raise ValueError('Download SHA256 mismatch; partial file retained')
            partial.rename(archive)
        prepare(archive, args.output)


if __name__ == '__main__':
    main()
