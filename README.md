# Local vLLM Batch Runner

Local inference code for the MEGC VQA experiments. The pipeline uses
motion/landmark features plus a frozen `Qwen2.5-7B-Instruct` model served
through vLLM.

## Setup

Run from the repository root:

```bash
cd <REPO_ROOT>
python -m pip install -e .
```

Download the MediaPipe Face Landmarker model to the repository root:

```bash
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -O face_landmarker.task
```

Official task documentation: [Google AI Edge Face Landmarker for Python](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python)

Download the reasoning checkpoint separately:

- `Qwen/Qwen2.5-7B-Instruct`
- <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>

## Expected Data

Use explicit paths in all commands below:

- `<DATA_ROOT>`: external MEGC data root
- `<TRAIN_JSONL>`: merged training VQA JSONL
- `<QWEN25_7B_INSTRUCT_DIR>`: local checkpoint directory

Expected layout under `<DATA_ROOT>`:

```text
<DATA_ROOT>/
  data/
    ME_VQA_MEGC_2025_Test_Crop_0329/
  trainset_flow/
    casme2/
    samm/
    testset/
```

## Export Features

Generate the clip-level calibration features before running the main pipeline:

```bash
python scripts/export_casme2_samm_motion_features.py \
  --mame-dir <DATA_ROOT> \
  --jsonl-path <TRAIN_JSONL> \
  --model-path <REPO_ROOT>/face_landmarker.task \
  --output-jsonl <REPO_ROOT>/outputs/casme2_samm_motion_features.jsonl \
  --output-csv <REPO_ROOT>/outputs/casme2_samm_motion_features.csv
```

## Run Validation

Full-prior CASME2:

```bash
local-vllm-batch-runner \
  --dataset casme2 \
  --mame-dir <DATA_ROOT> \
  --megc-jsonl <TRAIN_JSONL> \
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

Full-prior SAMM:

```bash
local-vllm-batch-runner \
  --dataset samm \
  --mame-dir <DATA_ROOT> \
  --megc-jsonl <TRAIN_JSONL> \
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

LOSO example:

```bash
local-vllm-batch-runner \
  --dataset samm \
  --loso \
  --loso-subject 10 \
  --mame-dir <DATA_ROOT> \
  --megc-jsonl <TRAIN_JSONL> \
  --llm_directory <QWEN25_7B_INSTRUCT_DIR>
```

## Included Here

- source code under `src/local_vllm_batch_runner/`
- feature export script under `scripts/`
- dependency metadata in `pyproject.toml`
- checkpoint note in `CHECKPOINT.md`
- result summary in `RESULTS_SUMMARY.md`
