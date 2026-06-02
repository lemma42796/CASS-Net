"""
Stage-wise runner for CASS-Net parameter sweeps.

Examples:
    python tools/run_cass_sweep.py --stage input_size --dry_run

    python tools/run_cass_sweep.py --stage topk \
        --opts DATASETS.ROOT_DIR /path/to/RGBNT201 \
               MODEL.PRETRAIN_PATH_T /path/to/vit.pth
"""
import argparse
import csv
import os
import re
import subprocess
import sys


STAGES = {
    'input_size': [
        ('input_256x128', 'configs/RGBNT201/sweeps/input_256x128.yml',
         'Input 256x128, 16x8 tokens'),
        ('input_384x192', 'configs/RGBNT201/sweeps/input_384x192.yml',
         'Input 384x192, 24x12 tokens'),
        ('input_192x384', 'configs/RGBNT201/sweeps/input_192x384.yml',
         'Input 192x384, 12x24 tokens'),
    ],
    'lr': [
        ('lr_5e_5', 'configs/RGBNT201/sweeps/lr_5e_5.yml',
         'Base LR 5e-5'),
        ('lr_1e_4', 'configs/RGBNT201/sweeps/lr_1e_4.yml',
         'Base LR 1e-4'),
        ('lr_2e_4', 'configs/RGBNT201/sweeps/lr_2e_4.yml',
         'Base LR 2e-4'),
    ],
    'warmup': [
        ('warmup_5', 'configs/RGBNT201/sweeps/warmup_5.yml',
         'Warmup 5 scheduler epochs'),
        ('warmup_10', 'configs/RGBNT201/sweeps/warmup_10.yml',
         'Warmup 10 scheduler epochs'),
        ('warmup_20', 'configs/RGBNT201/sweeps/warmup_20.yml',
         'Warmup 20 scheduler epochs'),
    ],
    'backbone_lr': [
        ('backbone_lr_0_05', 'configs/RGBNT201/sweeps/backbone_lr_0_05.yml',
         'Backbone LR factor 0.05'),
        ('backbone_lr_0_1', 'configs/RGBNT201/sweeps/backbone_lr_0_1.yml',
         'Backbone LR factor 0.1'),
        ('backbone_lr_0_2', 'configs/RGBNT201/sweeps/backbone_lr_0_2.yml',
         'Backbone LR factor 0.2'),
    ],
    'weight_decay': [
        ('weight_decay_0_02', 'configs/RGBNT201/sweeps/weight_decay_0_02.yml',
         'Weight decay 0.02'),
        ('weight_decay_0_05', 'configs/RGBNT201/sweeps/weight_decay_0_05.yml',
         'Weight decay 0.05'),
        ('weight_decay_0_1', 'configs/RGBNT201/sweeps/weight_decay_0_1.yml',
         'Weight decay 0.1'),
    ],
    'design': [
        ('cass_design_full', 'configs/RGBNT201/sweeps/cass_design_full.yml',
         'Quality-aware CASS + selected-token part + stable NGA'),
        ('cass_quality_off', 'configs/RGBNT201/sweeps/cass_quality_off.yml',
         'Disable CASS modality quality'),
        ('cass_part_off', 'configs/RGBNT201/sweeps/cass_part_off.yml',
         'Disable CASS selected-token part branch'),
        ('nga_no_warmup', 'configs/RGBNT201/sweeps/nga_no_warmup.yml',
         'Disable NGA warmup'),
        ('nga_no_ema', 'configs/RGBNT201/sweeps/nga_no_ema.yml',
         'Disable NGA memory EMA'),
        ('nga_no_prototype', 'configs/RGBNT201/sweeps/nga_no_prototype.yml',
         'Disable NGA identity prototypes'),
    ],
    'incremental': [
        ('baseline', 'configs/RGBNT201/sweeps/incremental_baseline.yml',
         'Baseline: shared ViT + simple token-summary fusion'),
        ('baseline_hss', 'configs/RGBNT201/sweeps/incremental_hss.yml',
         'Baseline + HSS'),
        ('baseline_hss_sqt', 'configs/RGBNT201/sweeps/incremental_hss_sqt.yml',
         'Baseline + HSS + SQT'),
        ('baseline_hss_sqt_nga', 'configs/RGBNT201/sweeps/incremental_hss_sqt_nga.yml',
         'Baseline + HSS + SQT + NGA'),
        ('baseline_hss_sqt_nga_cagf', 'configs/RGBNT201/sweeps/incremental_hss_sqt_nga_cagf.yml',
         'Baseline + HSS + SQT + NGA + CA-GF'),
    ],
    'nga_repair': [
        ('nga_repair_stable', 'configs/RGBNT201/sweeps/nga_repair_stable.yml',
         'A3 repair: additive SQT + stable NGA residual'),
        ('nga_cagf_repair_stable', 'configs/RGBNT201/sweeps/nga_cagf_repair_stable.yml',
         'A4 repair: additive SQT + stable NGA residual + CA-GF'),
    ],
    'nga_repair_v2': [
        ('nga_repair_query_sqt', 'configs/RGBNT201/sweeps/nga_repair_query_sqt.yml',
         'A3 repair v2: query-anchored NGA + gated SQT additive residual'),
    ],
    'cagf_repair_v2': [
        ('nga_cagf_repair_query_sqt', 'configs/RGBNT201/sweeps/nga_cagf_repair_query_sqt.yml',
         'A4 repair v2: residual CA-GF on query-anchored NGA + SQT additive'),
    ],
    'cagf_agree_v3': [
        ('nga_cagf_agree_query_sqt', 'configs/RGBNT201/sweeps/nga_cagf_agree_query_sqt.yml',
         'A4 repair v3: descriptor agreement CA-GF on query-anchored NGA + SQT additive'),
    ],
    'full_120sched_20stop': [
        ('cass_full_cagf_v3_120sched_20stop',
         'configs/RGBNT201/sweeps/full_cagf_v3_120sched_20stop.yml',
         'Full CASS v3: MAX_EPOCHS 120 schedule, stop after epoch 20'),
    ],
    'htl_exact_ablation_no_a0': [
        ('a1_hss',
         'configs/RGBNT201/sweeps/htl_exact_cass_a1_hss.yml',
         'A1: HTL-exact A0 + CASS HSS'),
        ('a2_hss_sqt',
         'configs/RGBNT201/sweeps/htl_exact_cass_a2_hss_sqt.yml',
         'A2: HTL-exact A0 + HSS + SQT additive'),
        ('a3_hss_sqt_nga',
         'configs/RGBNT201/sweeps/htl_exact_cass_a3_hss_sqt_nga.yml',
         'A3: HTL-exact A0 + HSS + SQT + NGA'),
        ('a4_hss_sqt_nga_cagf',
         'configs/RGBNT201/sweeps/htl_exact_cass_a4_hss_sqt_nga_cagf.yml',
         'A4: HTL-exact A0 + HSS + SQT + NGA + agreement CA-GF'),
    ],
    'topk': [
        ('topk_32', 'configs/RGBNT201/sweeps/topk_32.yml', 'CASS_TOPK 32'),
        ('topk_48', 'configs/RGBNT201/sweeps/topk_48.yml', 'CASS_TOPK 48'),
        ('topk_64', 'configs/RGBNT201/sweeps/topk_64.yml', 'CASS_TOPK 64'),
        ('topk_96', 'configs/RGBNT201/sweeps/topk_96.yml', 'CASS_TOPK 96'),
    ],
    'hss_edges': [
        ('hss_edges_128', 'configs/RGBNT201/sweeps/hss_edges_128.yml',
         'CASS_HSS_EDGES 128'),
        ('hss_edges_256', 'configs/RGBNT201/sweeps/hss_edges_256.yml',
         'CASS_HSS_EDGES 256'),
        ('hss_edges_384', 'configs/RGBNT201/sweeps/hss_edges_384.yml',
         'CASS_HSS_EDGES 384'),
    ],
    'hss_graph_weight': [
        ('hss_graph_weight_0_5', 'configs/RGBNT201/sweeps/hss_graph_weight_0_5.yml',
         'CASS_HSS_GRAPH_WEIGHT 0.5'),
        ('hss_graph_weight_1_0', 'configs/RGBNT201/sweeps/hss_graph_weight_1_0.yml',
         'CASS_HSS_GRAPH_WEIGHT 1.0'),
        ('hss_graph_weight_1_5', 'configs/RGBNT201/sweeps/hss_graph_weight_1_5.yml',
         'CASS_HSS_GRAPH_WEIGHT 1.5'),
    ],
    'nga_knn': [
        ('nga_knn_10', 'configs/RGBNT201/sweeps/nga_knn_10.yml', 'CASS_NGA_KNN 10'),
        ('nga_knn_20', 'configs/RGBNT201/sweeps/nga_knn_20.yml', 'CASS_NGA_KNN 20'),
        ('nga_knn_30', 'configs/RGBNT201/sweeps/nga_knn_30.yml', 'CASS_NGA_KNN 30'),
    ],
    'nga_refresh': [
        ('nga_refresh_1', 'configs/RGBNT201/sweeps/nga_refresh_1.yml',
         'Refresh NGA memory every epoch'),
        ('nga_refresh_2', 'configs/RGBNT201/sweeps/nga_refresh_2.yml',
         'Refresh NGA memory every 2 epochs'),
        ('nga_refresh_5', 'configs/RGBNT201/sweeps/nga_refresh_5.yml',
         'Refresh NGA memory every 5 epochs'),
    ],
    'selector': [
        ('selector_static_topk', 'configs/RGBNT201/sweeps/selector_static_topk.yml',
         'Static Top-K selector'),
        ('selector_shared_alpha', 'configs/RGBNT201/sweeps/selector_shared_alpha.yml',
         'Shared alpha across modalities'),
        ('selector_no_residual', 'configs/RGBNT201/sweeps/selector_no_residual.yml',
         'No soft residual for unselected tokens'),
        ('selector_full', 'configs/RGBNT201/sweeps/selector_full.yml',
         'Dynamic Top-K + modality alpha + soft residual'),
    ],
}

STAGE_ORDER = [
    'input_size',
    'lr',
    'warmup',
    'backbone_lr',
    'weight_decay',
    'incremental',
    'nga_repair',
    'nga_repair_v2',
    'cagf_repair_v2',
    'cagf_agree_v3',
    'full_120sched_20stop',
    'htl_exact_ablation_no_a0',
    'design',
    'topk',
    'hss_edges',
    'hss_graph_weight',
    'nga_knn',
    'nga_refresh',
    'selector',
]

_METRICS = [
    ('mAP', re.compile(r'Best Multi-Modal mAP:\s*([\d.]+)%')),
    ('Rank-1', re.compile(r'Best Multi-Modal Rank-1:\s*([\d.]+)%')),
    ('Rank-5', re.compile(r'Best Multi-Modal Rank-5:\s*([\d.]+)%')),
    ('Rank-10', re.compile(r'Best Multi-Modal Rank-10:\s*([\d.]+)%')),
]


def parse_log(log_path):
    out = {name: '' for name, _ in _METRICS}
    if not os.path.exists(log_path):
        return out
    with open(log_path, 'r') as f:
        for line in f:
            for name, pat in _METRICS:
                match = pat.search(line)
                if match:
                    out[name] = match.group(1)
    return out


def variants_for(stage):
    if stage == 'all':
        variants = []
        for stage_name in STAGE_ORDER:
            variants.extend((stage_name,) + item for item in STAGES[stage_name])
        return variants
    return [(stage,) + item for item in STAGES[stage]]


def run_variant(args, train_script, stage, key, overlay_yml, display):
    output_dir = os.path.join(args.output_dir, stage, key)
    cmd = [
        args.python,
        train_script,
        '--config_file', args.base,
        '--config_file', overlay_yml,
        'OUTPUT_DIR', output_dir,
    ] + args.opts
    print('\n' + '=' * 80)
    print('Stage:   {}'.format(stage))
    print('Variant: {}'.format(display))
    print('Output:  {}'.format(output_dir))
    print('Command: {}'.format(' '.join(cmd)))
    print('=' * 80)
    if args.dry_run:
        return 0, output_dir
    os.makedirs(output_dir, exist_ok=True)
    ret = subprocess.run(cmd, cwd=args.repo_root).returncode
    return ret, output_dir


def write_summary(rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'Variant', 'Overlay', 'Status'] + [name for name, _ in _METRICS])
        for row in rows:
            metrics = row['metrics']
            writer.writerow([
                row['stage'],
                row['key'],
                row['overlay'],
                row['status'],
            ] + [metrics[name] for name, _ in _METRICS])


def main():
    parser = argparse.ArgumentParser(description='Run CASS-Net stage-wise sweeps')
    parser.add_argument('--base', default='configs/RGBNT201/default.yml', type=str)
    parser.add_argument('--stage', default='input_size',
                        choices=['all'] + STAGE_ORDER)
    parser.add_argument('--output_dir', default='outputs/cass_sweeps/RGBNT201', type=str)
    parser.add_argument('--python', default=sys.executable, type=str)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--continue_on_error', action='store_true')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help='Extra cfg overrides passed to train_net.py')
    args = parser.parse_args()

    args.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_script = os.path.join(args.repo_root, 'train_net.py')
    rows = []

    for stage, key, overlay_yml, display in variants_for(args.stage):
        overlay_path = os.path.join(args.repo_root, overlay_yml)
        if not os.path.exists(overlay_path):
            raise FileNotFoundError(overlay_path)
        ret, output_dir = run_variant(args, train_script, stage, key, overlay_yml, display)
        status = 'dry-run' if args.dry_run else ('ok' if ret == 0 else 'failed:{}'.format(ret))
        rows.append({
            'stage': stage,
            'key': key,
            'overlay': overlay_yml,
            'status': status,
            'metrics': parse_log(os.path.join(output_dir, 'train_log.txt')),
        })
        if ret != 0 and not args.continue_on_error:
            break

    csv_path = os.path.join(args.output_dir, 'cass_sweep_summary.csv')
    write_summary(rows, csv_path)
    print('\nSummary CSV: {}'.format(csv_path))


if __name__ == '__main__':
    main()
