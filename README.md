# Local vLLM Batch Runner

This repository contains the local inference pipeline used for the MEGC VQA
experiments in the accompanying paper. The system replaces the remote Gemini
batch pipeline with a local, text-only reasoning stack built around motion
features, MediaPipe landmarks, and a frozen `Qwen2.5-7B-Instruct` model served
through vLLM.

## Project Layout

```text
MEGC2026/
  scripts/
    export_casme2_samm_motion_features.py
  src/local_vllm_batch_runner/
    batch_runner.py
    evaluation.py
    ...
  README.md
  pyproject.toml
  CHECKPOINT.md
  RESULTS_SUMMARY.md
```

## What This Repository Contains

- Unified batch runner for `CASME2`, `SAMM`, and the official `MEGC` test set
- Local feature extraction and calibration logic
- Evaluation script for UF1/UAR/BLEU/ROUGE
- Compact checkpoint and result notes for the accompanying submission

## Environment Requirements

- Python `>=3.10`
- Packages:
  - `mediapipe`
  - `numpy`
  - `pillow`
  - `transformers`
  - `vllm`
- MediaPipe asset:
  - `face_landmarker.task`
- Local reasoning checkpoint:
  - `Qwen/Qwen2.5-7B-Instruct`
  - Download: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>

Install the Python dependencies with:

```powershell
python -m pip install mediapipe numpy pillow transformers vllm
```

For local development, install the package in editable mode:

```powershell
python -m pip install -e .
```

## External Data Expected By The Runner

The code expects external MEGC data and precomputed optical-flow files. For a
clean reproduction, pass all paths explicitly instead of relying on defaults.

In the examples below, use these placeholders:

- `<REPO_ROOT>`
  - Local path to this repository
- `<DATA_ROOT>`
  - Local path to the external MEGC data root used by the runner
- `<TRAIN_JSONL>`
  - Training VQA JSONL file
- `<SAMPLE_ANSWER_DIR>`
  - Directory containing the official test-set answer templates
- `<QWEN25_7B_INSTRUCT_DIR>`
  - Local directory of the downloaded `Qwen2.5-7B-Instruct` checkpoint

Required inputs:

- `--mame-dir`
  - Root directory containing the cropped data and flow folders, referred to
    below as `<DATA_ROOT>`
- `--megc-jsonl`
  - Training VQA JSONL file, referred to below as `<TRAIN_JSONL>`
- `--sample-answer-dir`
  - Directory containing the official test-set answer templates, referred to
    below as `<SAMPLE_ANSWER_DIR>`
- `--llm_directory`
  - Local directory of the downloaded `Qwen2.5-7B-Instruct` checkpoint,
    referred to below as `<QWEN25_7B_INSTRUCT_DIR>`

The training/test assets are expected in the following layout under
`<DATA_ROOT>`:

```text
<DATA_ROOT>/
  data/
    ME_VQA_MEGC_2025_Test_Crop_0329/
  trainset_flow/
    casme2/
    samm/
    testset/
```

Place `face_landmarker.task` either:

- in this repository directory, or
- in the current working directory when running the scripts

## How To Run

Run all commands from this directory:

```powershell
cd <REPO_ROOT>
```

Install the package first:

```powershell
python -m pip install -e .
```

### 0. Export Motion Features For Calibration

Before running the main batch runner, generate the clip-level motion-feature
file used by the calibration logic:

```powershell
python scripts\export_casme2_samm_motion_features.py `
  --mame-dir <DATA_ROOT> `
  --jsonl-path <TRAIN_JSONL> `
  --model-path <REPO_ROOT>\face_landmarker.task `
  --output-jsonl <REPO_ROOT>\outputs\casme2_samm_motion_features.jsonl `
  --output-csv <REPO_ROOT>\outputs\casme2_samm_motion_features.csv
```

This step produces the calibration feature file expected by the runner:

- `outputs/casme2_samm_motion_features.jsonl`

### 1. Full-Prior Validation Runs

CASME2:

```powershell
local-vllm-batch-runner `
  --dataset casme2 `
  --mame-dir <DATA_ROOT> `
  --megc-jsonl <TRAIN_JSONL> `
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

SAMM:

```powershell
local-vllm-batch-runner `
  --dataset samm `
  --mame-dir <DATA_ROOT> `
  --megc-jsonl <TRAIN_JSONL> `
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

These runs produce:

- `outputs/batch_results_casme2.jsonl`
- `outputs/batch_results_samm.jsonl`
- `outputs/batch_results_vqa_casme2.jsonl`
- `outputs/batch_results_vqa_samm.jsonl`

### 2. LOSO Validation

CASME2 LOSO:

```powershell
local-vllm-batch-runner `
  --dataset casme2 `
  --loso `
  --mame-dir <DATA_ROOT> `
  --megc-jsonl <TRAIN_JSONL> `
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

SAMM LOSO:

```powershell
local-vllm-batch-runner `
  --dataset samm `
  --loso `
  --mame-dir <DATA_ROOT> `
  --megc-jsonl <TRAIN_JSONL> `
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

Optional single-subject LOSO:

```powershell
local-vllm-batch-runner `
  --dataset samm `
  --loso `
  --loso-subject 10 `
  --mame-dir <DATA_ROOT> `
  --megc-jsonl <TRAIN_JSONL> `
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

## Repository Scope

This GitHub repository is intended to stay focused on the runnable codebase and
the core documentation needed for verification.

Included here:

- the inference and evaluation code
- dependency metadata in `pyproject.toml`
- the main reproduction guide in `README.md`
- a compact checkpoint note in `CHECKPOINT.md`
- a compact result summary in `RESULTS_SUMMARY.md`

Large result dumps, submission JSONL files, and paper artifacts are better kept
in the supplementary ZIP rather than in the code repository itself.

## Notes

- The current local pipeline is text-only at inference time; the vision encoder
  is not used in the submitted configuration.
- The repository snapshot includes saved result files used to prepare the paper
  and supplementary materials.
- If you move the repository to a different workspace, prefer explicit path
  arguments for `--mame-dir`, `--megc-jsonl`, and `--sample-answer-dir`.
