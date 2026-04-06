#!/bin/bash

python -c "import torch; print('Version:', torch.__version__); print('Path:', torch.__file__)"

python -c "import torch; print('HIP available:', torch.cuda.is_available()); print('HIP version:', torch.version.hip)"
