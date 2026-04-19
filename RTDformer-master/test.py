from utils.device_utils import empty_cache, prepare_torch_runtime, resolve_device_config
prepare_torch_runtime()
import torch

print("PyTorch version:", torch.__version__)