# Benchmarking Population-Level Realism of LLM-based Society Simulation

## Abstract

Large language models (LLMs) show great promise for simulating human societies, potentially expanding the methodological toolkit of computational social science.
However, while existing studies have examined the individual-level predictability or behavioral plausibility of LLM simulations, their ability to reproduce real-world population-level social statistical patterns—which lie at the core of social science research—remains largely untested. 
This work presents the first systematic benchmark of population-level realism in LLM-based society simulations, evaluating how well simulated populations reproduce key statistical patterns observed in real societies.
 We analyze five types of statistical patterns central to social science research, including univariate distributions, bivariate associations, multivariate prediction of social outcomes, life-course event sequence distributions, and multivariate prediction of life-course event sequences.
 Our benchmark covers 10 longitudinal and 5 cross-sectional datasets encompassing xx variables across six major social domains: demographics, socioeconomic status (SES), marriage, health, abilities, and attitudes.
Our benchmark establishes a unified statistical testbed for quantifying population-level realism, laying the foundation for rigorous evaluation of AI-driven society simulations.

## Setup

1. Install dependencies.

2. Create `.env` with your OpenAI key if you plan to regenerate simulations:

   ```bash
   echo "OPENROUTER_API_KEY=your_key" > .env
   # Or use OpenAI api
   echo "OPENAI_API_KEY=your_key" > .env
   ```

## Data Preparation

- Place **real datasets** under `real_data/` (e.g., `real_data/nlsy1023.csv`).
- Place **simulated datasets** under `simulated_data/` (e.g., `simulated_data/<run_name>/sim_profiles_*.csv`).
- Evaluation outputs are written to `evaluation_results/` by the runners.

## Simulation

Current pipeline in `simulation/` with configs under `simulation/configs/{content,param}`. It enforces JSON responses via OpenRouter, retries malformed rows, and writes each run to `simulated_data/<run_name>/` alongside `failed_responses/`.

Common entrypoints:
```bash
# Dataset-specific runners (override SIM_* envs as needed)
bash scripts/simulation/acs_1980.sh
bash scripts/simulation/cfps.sh
bash scripts/simulation/cps_1980.sh
bash scripts/simulation/gss_2018.sh
bash scripts/simulation/nlsy79.sh
bash scripts/simulation/addhealth.sh

# Single-model smoketests
bash scripts/simulation/gpt5_sanity.sh
bash scripts/simulation/gemini_sanity.sh
bash scripts/simulation/qwen_sanity.sh
bash scripts/simulation/claude_sanity.sh
bash scripts/simulation/deepseek_sanity.sh
bash scripts/simulation/grok_sanity.sh

# Run all model smoketests sequentially
bash scripts/simulation/split_models/all_models_sanity.sh
```

Environment overrides respected by all scripts: `NUM_PROFILES`, `MAX_WORKERS`, `SIM_CONTENT_CONFIG`, `SIM_PARAMS_CONFIG`, `SIM_OUTPUT_DIR`.

## Evaluation

### Metrics
- **Type1**: Single-variable distribution equivalence (bootstrap insignificance rate).
- **Type2**: Pairwise association strength equivalence (categorical/numeric mixes; structure + strength).
- **Type3**: Regression coefficient stability (OLS/Logit/MNLogit, bootstrap).
- **Type4**: Event order distribution similarity (life-course sequencing).
- **Type5**: Event order regression stability.

### Usage

Entry scripts (one per dataset; outputs default to `evaluation_results/<dataset>/...`):
```bash
python scripts/evaluation/acs_1980.py
python scripts/evaluation/cfps.py
python scripts/evaluation/nlsy.py
python scripts/evaluation/addhealth.py
python scripts/evaluation/gss_2018.py
python scripts/evaluation/cps_1980.py \
  --sim-root /path/to/simulated_data/cps_1980 \
  --output-base ./evaluation_results/cps_1980 \
  --config ./evaluation/config/cps_1980/evaluation_master.yaml
```
All scripts accept the same overrides: `--sim-root`, `--output-base`, `--config`, `--sampled-prefix`, `--sim-prefix`, `--eval-script`. They default to batch mode (scan subfolders for `sampled_*.csv` / `sim_*.csv` pairs). To evaluate a single run directory, pass `--single` and point `--sim-root` at that run (e.g., `--sim-root simulated_data/nlsy/<run_dir> --single`).

## Visualization

Plots live in `visualization/` (code) and `scripts/visualization/` (entrypoints). Figures are written to `visualization_figures/`.

### Overview heatmaps & distributions
```bash
# Traverse datasets under evaluation_results (default) and plot all runs
bash scripts/visualization/run_overview.sh

# Point to a specific dataset folder and only plot that folder's CSVs
bash scripts/visualization/run_overview.sh evaluation_results/nlsy --single
```
Flags accepted by `visualization/overview.py` (passed through by the script):
- `--root` (positional via the shell script) sets the evaluation results root. Default `evaluation_results`.
- `--single` treats `--root` as a single run/dataset folder instead of scanning subfolders.

### Strength vs structure (Types 2/3/5)
```bash
# All datasets under the root
bash scripts/visualization/run_structure_vs_strength.sh

# Single dataset folder (no subfolder scan)
bash scripts/visualization/run_structure_vs_strength.sh evaluation_results/nlsy --single

# Limit to specific datasets (comma-separated) in batch mode
bash scripts/visualization/run_structure_vs_strength.sh --datasets nlsy,acs_1980
```
Flags accepted by `visualization/structure_vs_strength.py`:
- `--root` (positional via the shell script) sets the evaluation results root. Default `evaluation_results`.
- `--datasets` to limit which dataset subfolders are scanned (batch mode only).
- `--single` treats `--root` as one run/dataset folder.
