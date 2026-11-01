"""Rescore existing MF-Clean predictions; no training or model inference.

Run with the CPU requirements installed:
  python run_baseline.py --repo /path/to/MedForgePlusPlus --out /path/to/results
"""
import argparse
import collections
import csv
import hashlib
import json
import platform
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=2027)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / 'v3'))
    import stats_v3 as stats
    import numpy as np

    files = {
        'initial_sft_cf': 'gen_test_v3_sft_cf.jsonl',
        'balanced_continued_sft_cf': 'gen_test_contsft_cf_bal.jsonl',
        'image_grpo_cf': 'gen_test_ct_image_1697.jsonl',
    }
    inputs, records, raw_rows, independent = {}, {}, {}, {}
    for name, filename in files.items():
        path = args.repo / 'results/predictions' / filename
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        by_image = {r['image']: r for r in raw}
        assert len(by_image) == len(raw), f'Duplicate prediction identifiers: {name}'
        raw_rows[name] = by_image
        records[name] = stats.load_gen(str(path))
        inputs[name] = {'file': str(path.relative_to(args.repo)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        n, correct = collections.Counter(), collections.Counter()
        for row in raw:
            gt = 'real' if row['type'] == 'real' else 'deepfake'
            verdicts = re.findall(r'<verdict>\s*(real|deepfake)\s*</verdict>', row['output'], re.I)
            assert len(verdicts) == 1, 'Independent parser expects one explicit verdict'
            n[gt] += 1
            correct[gt] += int(verdicts[0].lower() == gt)
        independent[name] = {'counts': dict(n), 'correct': dict(correct),
                             'acc': sum(correct.values()) / len(raw),
                             'bal_acc': sum(correct[c] / n[c] for c in n) / 2}

    reference = raw_rows['initial_sft_cf']
    for name, rows in raw_rows.items():
        assert rows.keys() == reference.keys(), f'Unequal evaluated images: {name}'
        for image, row in rows.items():
            for field in ['type', 'group', 'organ', 'box', 'twin', 'orig']:
                assert row.get(field) == reference[image].get(field), (name, image, field)
    groups = {r['group'] for r in reference.values()}
    assert len(reference) == 2756 and len(groups) == 1008
    fake_rows = [r for r in reference.values() if r['type'] != 'real']
    assert all(r.get('box') for r in fake_rows)
    for row in fake_rows:
        twin = row.get('twin')
        if twin is not None:
            assert twin in reference
            assert reference[twin]['type'] == 'real'
            assert reference[twin]['group'] == row['group']

    for name, verdict in [('always_real', 'real'), ('always_fake', 'deepfake')]:
        raw = []
        for original in reference.values():
            row = dict(original)
            row['output'] = f'<verdict>{verdict}</verdict>'
            raw.append(stats.normalize(row))
        records[name] = stats.attach_twins(raw)

    summaries = {name: stats.summarize(rows, B=args.bootstrap, seed=args.seed)
                 for name, rows in records.items()}
    for name, tally in independent.items():
        for metric in ['acc', 'bal_acc']:
            assert abs(summaries[name]['point'][metric] - tally[metric]) < 1e-12
    comparisons = {}
    for a, b in [('initial_sft_cf', 'balanced_continued_sft_cf'),
                 ('initial_sft_cf', 'image_grpo_cf'),
                 ('image_grpo_cf', 'balanced_continued_sft_cf')]:
        comparisons[f'{b}_minus_{a}'] = stats.paired(records[a], records[b],
                                                    B=args.bootstrap, seed=args.seed)
    per_organ = {name: stats.breakdown(rows, 'organ', args.bootstrap, args.seed)
                 for name, rows in records.items() if name in files}
    report = {
        'operation': 'CPU rescore of existing published predictions; no training or inference',
        'hypothesis': 'Does balanced continued SFT improve over initial CF SFT on the fixed MF-Clean test?',
        'primary_metric': 'binary balanced accuracy',
        'commit': subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__},
        'bootstrap': {'unit': 'declared source group', 'replicates': args.bootstrap, 'seed': args.seed},
        'inputs': inputs, 'integrity_checks': {'same_images_and_metadata': True,
            'no_duplicate_identifiers': True, 'independent_verdict_tally_matches': True,
            'images': len(reference), 'groups': len(groups), 'raw_images_verified': False},
        'independent_verdict_tallies': independent, 'summaries': summaries,
        'paired': comparisons, 'per_organ': per_organ,
        'limitations': ['Existing runs, not a new training replication',
            'No raw data, weights or original training manifests available locally',
            'Intervals condition on fixed models and declared groups, not seed variability',
            'Balanced-run training recipe and matched compute budget are not established'],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    metrics = ['acc', 'bal_acc', 'f1_macro', 'rec_real', 'rec_implant', 'rec_removal', 'mIoU', 'twin']
    with (args.out / 'metrics_percent.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['model'] + metrics)
        for name, summary in summaries.items():
            writer.writerow([name] + [100 * summary['point'][k] for k in metrics])
    print((args.out / 'metrics_percent.csv').read_text())
    for name, comparison in comparisons.items():
        diff = comparison['diff']['bal_acc']
        print(name, 'BA difference (pp):', 100 * diff['B_minus_A'],
              '95% CI:', [100 * x for x in diff['ci95']])
    print('Saved:', args.out / 'results.json')


if __name__ == '__main__':
    main()
