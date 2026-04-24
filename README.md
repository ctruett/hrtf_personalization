# applsci-08-02180

Modern reimplementation scaffold for:

Lee, G.W.; Kim, H.K. "Personalized HRTF Modeling Based on Deep Neural Network Using Anthropometric Measurements and Images of the Ear." *Applied Sciences* 2018, 8(11), 2180.

This project keeps two tracks:

- `baseline`: paper-faithful reproduction of the original three-subnetwork model and leave-one-subject-out evaluation.
- `conditional`: a modernized single-model path that conditions on source direction instead of training one model per direction.

## Stack

- Python 3.9+
- PyTorch 2.x
- `sofar` for SOFA I/O and validation
- `pyfar` for signal-aware audio processing support

## Layout

```text
configs/                YAML configs for dataset, model, training, export
src/hrtf_personalization/
  cli/                  command entrypoints
  data/                 dataset loading and typed manifests
  preprocessing/        anthropometrics and ear-image preprocessing
  models/               baseline and conditional PyTorch models
  training/             training loops and split generation
  evaluation/           objective metrics and evaluation runners
  sofa/                 SOFA export helpers
  rendering/            placeholder binaural rendering helpers
tests/                  smoke tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
hrtf-personalization --help
```

## Recommended Workflow

1. Clone the source dataset repo:
```bash
git clone https://github.com/codyjhsieh/HRTFCNN.git
```

2. Fetch external assets using the same sources used in `FinalHRTFCNN.ipynb`:
```bash
hrtf-personalization fetch-hrtfcnn-assets --config configs/dataset.yaml
```

3. Prepare notebook-compatible metadata and ear-image artifacts:
```bash
hrtf-personalization prepare-cipic --config configs/dataset.yaml
```

4. Train the paper-faithful baseline:
```bash
hrtf-personalization train-baseline --config configs/train-baseline.yaml
```

5. Train the notebook-style conditional model:
```bash
hrtf-personalization train-conditional --config configs/train-conditional.yaml
```

6. Run evaluation (model type is auto-detected from checkpoint):
```bash
hrtf-personalization evaluate --config configs/eval-baseline.yaml
hrtf-personalization evaluate --config configs/eval-conditional.yaml
```

7. Export predicted HRTFs to SOFA:
```bash
hrtf-personalization export-sofa --config configs/export.yaml
```

To run inference on a new ear image:
```bash
hrtf-personalization predict \
  --checkpoint artifacts/checkpoints/conditional.pt \
  --image /path/to/ear.jpg \
  --template-sofa /tmp/HRTFCNN/data/template.sofa
```

If `--output-sofa` is omitted, the CLI writes a timestamped file to `predictions/`, for example
`predictions/prediction-20260423-223500.sofa`.

If you have both ear photos, you can provide them explicitly instead of mirroring a single image:
```bash
hrtf-personalization predict \
  --checkpoint artifacts/checkpoints/conditional.pt \
  --left-image /path/to/left-ear.jpg \
  --right-image /path/to/right-ear.jpg \
  --template-sofa /tmp/HRTFCNN/data/template.sofa
```

To make left/right prediction meaningful, rerun `prepare-cipic` and retrain after updating to the
ear-side-aware model. The preparation step now writes both the photographed ear and a mirrored
counterpart so the model can learn separate left/right outputs from one available ear photo.

If you want the notebook-style interactive anthropometric workflow first:
```bash
hrtf-personalization measure-anthro \
  --front-image /path/to/front.jpg \
  --side-image /path/to/side.jpg \
  --output artifacts/predictions/my_subject.anthro.json

hrtf-personalization predict \
  --checkpoint artifacts/checkpoints/conditional.pt \
  --image /path/to/ear.jpg \
  --anthro-json artifacts/predictions/my_subject.anthro.json \
  --template-sofa /tmp/HRTFCNN/data/template.sofa
```

## Current Status

This repository is a working scaffold. It includes package structure, typed interfaces, CLI wiring,
model definitions, metric functions, SOFA export helpers, and a `prepare-cipic` command aligned to
the HRTFCNN notebook dataset layout. It also includes `fetch-hrtfcnn-assets`, which follows the
notebook asset acquisition flow (`wget` CIPIC SOFA + Google Drive archives).
