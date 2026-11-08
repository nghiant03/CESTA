# CESTA

Communication-Efficient Spatial-Temporal Aggregation for sensor-network fault diagnosis.

## Setup and workflow

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nghiant03/CESTA.git
cd CESTA
uv sync
uv run cesta --help
```

Transform, train, then evaluate a validation-selected checkpoint:

```bash
uv run cesta transform intel_lab data/raw/Intel/data.txt data/datasets/Intel_fault15 \
  --config config/datasets/intel-lab/fault-15.yaml
uv run cesta train config/training/Intel/cesta.yaml data/datasets/Intel_fault15
uv run cesta evaluate --model runs/cesta/<run_id> --data data/datasets/Intel_fault15
```

## Experiment commands

Run scripts with `uv run python`; inspect supported arguments with `--help`. Experiment output directories must be new. Preserve protocol hashes, resolved configs, interrupted attempts, and historical seed 1242 artifacts. Do not select recipes, fallback candidates, or replacement seeds from test scores.

```bash
uv run python scripts/run_all_baselines.py --dry-run
uv run python scripts/run_all_baselines.py --num-gpus 4 --keep-progress-log
uv run python scripts/audit_decisive_comparison.py \
  --spec config/benchmarks/decisive-comparison.yaml \
  --runs-root runs --output runs/decisive-comparison-audit
uv run python scripts/summarize_decisive_comparison.py \
  --runs-csv runs/decisive-comparison-audit/runs.csv \
  --output runs/decisive-comparison-summary \
  --comparison <variant> <locked-reference>
uv run cesta optimize --data data/datasets/Intel_fault15 --model cnn1d --n-trials 20 --epochs 10
```

## Repository layout

| Path | Contents |
| --- | --- |
| [`src/CESTA/`](src/CESTA/) | Datasets, fault injection, models, training, evaluation, CLI |
| [`config/datasets/`](config/datasets/) | Dataset transformation and fault-injection configs |
| [`config/training/`](config/training/) | Default model training configs |
| [`config/experiments/`](config/experiments/) | Intel comparison variants and supporting DHT training recipes |
| [`config/benchmarks/`](config/benchmarks/) | Retained Intel comparison specification |
| [`scripts/`](scripts/) | Benchmark runners, audits, analysis, firmware export |
| [`firmware/`](firmware/) | ESP32-S3 Rust application |
| `data/` | Raw data and transformed datasets |
| `runs/` | Generated checkpoints, metrics, and metadata |

## License

[MIT](LICENSE)
