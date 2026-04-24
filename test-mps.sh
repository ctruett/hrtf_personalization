#!/bin/bash
source .venv/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print("mps_built:", torch.backends.mps.is_built())
print("mps_available:", torch.backends.mps.is_available())
PY
