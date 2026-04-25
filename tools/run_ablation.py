"""
One-click runner for the HTL ablation study.

For each variant, invokes train_net.py with the base dataset config plus
an ablation overlay yml that flips the AGF / HSL switches. After all runs
finish, parses train_log.txt from each output dir and prints / writes a
summary CSV.

Example:
    python tools/run_ablation.py \\
        --base configs/RGBNT201/EDITOR.yml \\
        --ablation_dir configs/RGBNT201/ablations \\
        --output_dir ablation_results
"""
import argparse
import csv
import os
import re
import subprocess
import sys


# Order here controls the variants run and the row order in the summary.
VARIANTS = [
    ('baseline', 'baseline.yml', 'Baseline (HS+FACSS only)'),
    ('hsl_only', 'hsl_only.yml', '+HSL'),
    ('agf_only', 'agf_only.yml', '+AGF'),
    ('full',     'full.yml',     'Full (AGF+HSL)'),
]


_METRICS = [
    ('mAP',    re.compile(r'Best Multi-Modal mAP:\s*([\d.]+)%')),
    ('Rank-1', re.compile(r'Best Multi-Modal Rank-1:\s*([\d.]+)%')),
    ('Rank-5', re.compile(r'Best Multi-Modal Rank-5:\s*([\d.]+)%')),
    ('Rank-10', re.compile(r'Best Multi-Modal Rank-10:\s*([\d.]+)%')),
]


def parse_log(log_path):
    out = {name: 0.0 for name, _ in _METRICS}
    if not os.path.exists(log_path):
        return out
    with open(log_path, 'r') as f:
        for line in f:
            for name, pat in _METRICS:
                m = pat.search(line)
                if m:
                    out[name] = float(m.group(1))
    return out


def run_variant(train_script, base_yml, overlay_yml, output_dir, python_exe):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        python_exe, train_script,
        '--config_file', base_yml,
        '--config_file', overlay_yml,
        'OUTPUT_DIR', output_dir,
    ]
    print('Running: {}'.format(' '.join(cmd)))
    return subprocess.run(cmd, cwd=os.path.dirname(train_script)).returncode


def main():
    parser = argparse.ArgumentParser(description="HTL ablation runner")
    parser.add_argument("--base", required=True, type=str,
                        help="Base dataset yml (e.g. configs/RGBNT201/EDITOR.yml)")
    parser.add_argument("--ablation_dir", required=True, type=str,
                        help="Directory containing baseline/hsl_only/agf_only/full.yml")
    parser.add_argument("--output_dir", default="ablation_results", type=str)
    parser.add_argument("--python", default=sys.executable, type=str)
    args = parser.parse_args()

    train_script = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'train_net.py'))

    results = {}
    for key, overlay_name, display in VARIANTS:
        overlay_yml = os.path.join(args.ablation_dir, overlay_name)
        if not os.path.exists(overlay_yml):
            print('SKIP {}: missing {}'.format(key, overlay_yml))
            continue
        exp_dir = os.path.join(args.output_dir, key)
        print('\n' + '=' * 60)
        print('Ablation: {}'.format(display))
        print('Output:   {}'.format(exp_dir))
        print('=' * 60)
        ret = run_variant(train_script, args.base, overlay_yml, exp_dir, args.python)
        if ret != 0:
            print('WARNING: {} exited with code {}'.format(key, ret))
        results[key] = parse_log(os.path.join(exp_dir, 'train_log.txt'))

    print('\n' + '=' * 60)
    print('Ablation Results Summary')
    print('=' * 60)
    header = '{:<28s}'.format('Variant') + ' '.join('{:>8s}'.format(n) for n, _ in _METRICS)
    print(header)
    print('-' * len(header))
    for key, _, display in VARIANTS:
        if key not in results:
            continue
        r = results[key]
        print('{:<28s}'.format(display) + ' '.join(
            '{:>7.2f}%'.format(r[n]) for n, _ in _METRICS))

    csv_path = os.path.join(args.output_dir, 'ablation_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Variant'] + [n for n, _ in _METRICS])
        for key, _, display in VARIANTS:
            if key not in results:
                continue
            r = results[key]
            writer.writerow([display] + ['{:.2f}'.format(r[n]) for n, _ in _METRICS])
    print('\nCSV saved to {}'.format(csv_path))


if __name__ == '__main__':
    main()
