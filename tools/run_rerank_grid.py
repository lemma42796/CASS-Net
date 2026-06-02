"""
Run a small re-ranking grid for an existing checkpoint.

Example:
    python tools/run_rerank_grid.py \
        --config_file configs/RGBNT201/default.yml \
        --config_file configs/RGBNT201/sweeps/cass_hss_sqt_quality_cls.yml \
        --checkpoint /path/to/CASS-Net_best.pth \
        --output_dir /path/to/rerank_grid \
        --k1 40,45,50,55 \
        --k2 15,20 \
        --lambda_values 0.08,0.10,0.12,0.15 \
        --opts DATASETS.ROOT_DIR /path/to/datasets
"""
import argparse
import os
import re
import subprocess
import sys


_METRICS = [
    ("mAP", re.compile(r"\bmAP:\s*([\d.]+)%")),
    ("Rank-1", re.compile(r"Rank-1\s*:?\s*([\d.]+)%")),
    ("Rank-5", re.compile(r"Rank-5\s*:?\s*([\d.]+)%")),
    ("Rank-10", re.compile(r"Rank-10\s*:?\s*([\d.]+)%")),
]


def parse_float_list(raw):
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw):
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_metrics(log_path):
    out = {name: "" for name, _ in _METRICS}
    if not os.path.exists(log_path):
        return out
    with open(log_path, "r") as handle:
        for line in handle:
            for name, pattern in _METRICS:
                match = pattern.search(line)
                if match:
                    out[name] = match.group(1)
    return out


def as_sort_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def write_summary(rows, output_dir):
    header = ["k1", "k2", "lambda", "status"] + [name for name, _ in _METRICS]
    summary_path = os.path.join(output_dir, "summary.tsv")
    sorted_path = os.path.join(output_dir, "summary_sorted.tsv")

    with open(summary_path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            values = [
                str(row["k1"]),
                str(row["k2"]),
                "{:.3g}".format(row["lambda"]),
                row["status"],
            ] + [row["metrics"][name] for name, _ in _METRICS]
            handle.write("\t".join(values) + "\n")

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            as_sort_float(row["metrics"]["mAP"]),
            as_sort_float(row["metrics"]["Rank-1"]),
        ),
        reverse=True,
    )
    with open(sorted_path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for row in sorted_rows:
            values = [
                str(row["k1"]),
                str(row["k2"]),
                "{:.3g}".format(row["lambda"]),
                row["status"],
            ] + [row["metrics"][name] for name, _ in _METRICS]
            handle.write("\t".join(values) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run TEST.RERANK_* grid")
    parser.add_argument(
        "--config_file",
        action="append",
        default=[],
        required=True,
        help="Config file passed to test_net.py; repeat to chain merges.",
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--python", default=sys.executable, type=str)
    parser.add_argument("--k1", default="40,45,50,55", type=str)
    parser.add_argument("--k2", default="15,20", type=str)
    parser.add_argument("--lambda_values", default="0.08,0.10,0.12,0.15", type=str)
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra cfg overrides passed to test_net.py",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_script = os.path.join(repo_root, "test_net.py")
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for k1 in parse_int_list(args.k1):
        for k2 in parse_int_list(args.k2):
            for lam in parse_float_list(args.lambda_values):
                lam_tag = str(lam).replace(".", "p")
                exp_dir = os.path.join(args.output_dir, "k1_{}_k2_{}_l_{}".format(k1, k2, lam_tag))
                os.makedirs(exp_dir, exist_ok=True)
                cmd = [args.python, test_script]
                for config_path in args.config_file:
                    cmd.extend(["--config_file", config_path])
                cmd.extend([
                    "OUTPUT_DIR",
                    exp_dir,
                    "TEST.WEIGHT",
                    args.checkpoint,
                    "TEST.RERANK_K1",
                    str(k1),
                    "TEST.RERANK_K2",
                    str(k2),
                    "TEST.RERANK_LAMBDA",
                    str(lam),
                ])
                cmd.extend(args.opts)

                stdout_path = os.path.join(exp_dir, "stdout.log")
                print("Running {}".format(" ".join(cmd)), flush=True)
                status = "dry_run"
                if not args.dry_run:
                    with open(stdout_path, "w") as stdout:
                        ret = subprocess.run(
                            cmd,
                            cwd=repo_root,
                            stdout=stdout,
                            stderr=subprocess.STDOUT,
                        ).returncode
                    status = "ok" if ret == 0 else "exit_{}".format(ret)
                metrics = parse_metrics(stdout_path)
                rows.append({
                    "k1": k1,
                    "k2": k2,
                    "lambda": lam,
                    "status": status,
                    "metrics": metrics,
                })
                write_summary(rows, args.output_dir)
                if status != "ok" and not args.continue_on_error and not args.dry_run:
                    return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
