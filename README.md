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

The optical-flow preprocessing uses OpenCV TV-L1 via `opencv-contrib-python`.

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
- `<SAMM_ROOT>`: raw SAMM cropped-frame root
- `<CASME2_ROOT>`: raw CASME II frame root
- `<SAMM_ANNO_XLSX>`: SAMM annotation spreadsheet
- `<CASME2_ANNO_XLSX>`: CASME II annotation spreadsheet
- `<TESTSET_CROP_ROOT>`: cropped test-set root containing `selected_frames/`
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

## Prepare Optical Flow

Generate trainset TV-L1 flow in the directory layout expected by the runner:

```bash
python scripts/generate_trainset_flow.py \
  --jsonl <TRAIN_JSONL> \
  --samm-root <SAMM_ROOT> \
  --casme2-root <CASME2_ROOT> \
  --samm-anno <SAMM_ANNO_XLSX> \
  --casme2-anno <CASME2_ANNO_XLSX> \
  --output-root <DATA_ROOT>/trainset_flow
```

Generate test-set flow from the cropped `selected_frames/` layout:

```bash
python scripts/generate_testset_flow_from_crops.py \
  --input <TESTSET_CROP_ROOT> \
  --output-root <DATA_ROOT>/trainset_flow/testset \
  --image-size 224
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
