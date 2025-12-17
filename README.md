# Evaluating Statistical Realism of LLM-generated Social Science Data

## Abstract
Large Language Models (LLMs) show great promise for generating social science data, potentially expanding the methodological toolkit of computational social science.
However, while existing studies have examined the individual-level predictability or behavioral plausibility of LLM simulations, their ability to reproduce real-world, population-level statistical patterns, which lie at the core of social science research, remains largely untested. 
In this article, we introduce SSDataBench, the first systematic benchmark designed to evaluate population-level statistical realism in LLM-generated social science data. The benchmark assesses five types of statistical patterns central to social research: univariate distributions, bivariate associations, multivariate outcome predictions, life event sequence distributions, and the associations between life event sequences and covariate variables. SSDataBench spans four longitudinal datasets and three cross-sectional datasets across six major social domains: demographics, socioeconomic status, marriage, health, abilities, and attitudes.
Our analysis uncovers systematic representational limitations in current LLMs, which manifest as a marked tendency to compress real-world heterogeneity into simplified topological structures.
Overall, this work establishes a unified statistical testbed for assessing population-level realism and highlights the need for models that represent heterogeneous human societies with greater statistical fidelity.
## Setup

1. Install dependencies.

  ```bash
  pip install -r requirements.txt
  ```

2. Create `.env` with your API key if you plan to regenerate simulations:

   ```bash
   # Use OpenRouter api
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
bash scripts/simulation/understandingsociety.sh

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

### Usage

Entry scripts (one per dataset; outputs default to `evaluation_results/<dataset>/...`):
```bash
python scripts/evaluation/acs_1980.py
python scripts/evaluation/cfps.py
python scripts/evaluation/nlsy.py
python scripts/evaluation/addhealth.py
python scripts/evaluation/gss_2018.py
python scripts/evaluation/cps_1980.py 
python scripts/evaluation/us.py \
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
```
Flags accepted by `visualization/overview.py` (passed through by the script):
- `--root` (positional via the shell script) sets the evaluation results root. Default `evaluation_results`.
- `--single` treats `--root` as a single run/dataset folder instead of scanning subfolders.

