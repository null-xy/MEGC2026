# Local vLLM Batch Runner

This repository contains the local inference pipeline used for the MEGC VQA
experiments in the accompanying paper. The system replaces the remote Gemini
batch pipeline with a local, text-only reasoning stack built around motion
features, MediaPipe landmarks, and a frozen `Qwen2.5-7B-Instruct` model served
through vLLM.

## What This Repository Contains

- Unified batch runner for `CASME2`, `SAMM`, and the official `MEGC` test set
- Local feature extraction and calibration logic
- Evaluation script for UF1/UAR/BLEU/ROUGE
- Saved validation metrics and leaderboard JSONL outputs used for submission

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

## External Data Expected By The Runner

The code expects external MEGC data and precomputed optical-flow files. For a
clean reproduction, pass all paths explicitly instead of relying on defaults.

Required inputs:

- `--mame-dir`
  - Root directory containing the cropped data and flow folders
- `--megc-jsonl`
  - Training VQA JSONL file
- `--sample-answer-dir`
  - Directory containing the official test-set answer templates
- `--llm_directory`
  - Local directory of the downloaded `Qwen2.5-7B-Instruct` checkpoint

The training/test assets are expected in the following layout under
`--mame-dir`:

```text
<mame-dir>/
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
cd D:\thesis\mame\local_vllm_batch_runner
```

### 1. Full-Prior Validation Runs

CASME2:

```powershell
python batch_runner.py `
  --dataset casme2 `
  --mame-dir <PATH_TO_MAME_ROOT> `
  --megc-jsonl <PATH_TO_TRAIN_JSONL> `
  --llm_directory <PATH_TO_QWEN25_7B_INSTRUCT>
```

SAMM:

```powershell
python batch_runner.py `
  --dataset samm `
  --mame-dir <PATH_TO_MAME_ROOT> `
  --megc-jsonl <PATH_TO_TRAIN_JSONL> `
  --llm_directory <PATH_TO_QWEN25_7B_INSTRUCT>
```

These runs produce:

- `outputs/batch_results_casme2.jsonl`
- `outputs/batch_results_samm.jsonl`
- `outputs/batch_results_vqa_casme2.jsonl`
- `outputs/batch_results_vqa_samm.jsonl`

### 2. LOSO Validation

CASME2 LOSO:

```powershell
python batch_runner.py `
  --dataset casme2 `
  --loso `
  --mame-dir <PATH_TO_MAME_ROOT> `
  --megc-jsonl <PATH_TO_TRAIN_JSONL> `
  --llm_directory <PATH_TO_QWEN25_7B_INSTRUCT>
```

SAMM LOSO:

```powershell
python batch_runner.py `
  --dataset samm `
  --loso `
  --mame-dir <PATH_TO_MAME_ROOT> `
  --megc-jsonl <PATH_TO_TRAIN_JSONL> `
  --llm_directory <PATH_TO_QWEN25_7B_INSTRUCT>
```

Optional single-subject LOSO:

```powershell
python batch_runner.py `
  --dataset samm `
  --loso `
  --loso-subject 10 `
  --mame-dir <PATH_TO_MAME_ROOT> `
  --megc-jsonl <PATH_TO_TRAIN_JSONL> `
  --llm_directory <PATH_TO_QWEN25_7B_INSTRUCT>
```

### 3. Official Test-Set Submission Files

```powershell
python batch_runner.py `
  --dataset testset `
  --mame-dir <PATH_TO_MAME_ROOT> `
  --sample-answer-dir <PATH_TO_SAMPLE_ANSWER_DIR> `
  --llm_directory <PATH_TO_QWEN25_7B_INSTRUCT>
```

This generates the leaderboard submission files:

- `me_vqa_casme3_v2_test_pred.jsonl`
- `me_vqa_samm_v2_test_pred.jsonl`

## Evaluation

To recompute the validation metrics from the saved validation VQA outputs:

```powershell
python evaluation.py `
  --input outputs\batch_results_vqa_casme2.jsonl `
  --input outputs\batch_results_vqa_samm.jsonl `
  --json-out outputs\vqa_eval_metrics.json
```

The evaluation script reports:

- UF1 Coarse
- UAR Coarse
- UF1 Fine-Grained
- UAR Fine-Grained
- BLEU
- ROUGE

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
