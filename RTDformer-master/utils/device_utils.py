import ctypes
import gc
import os
import site
import sys
from contextlib import nullcontext
from pathlib import Path


def _dedupe_paths(paths):
    unique_paths = []
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        key = str(resolved).lower()
        if key not in seen and resolved.exists():
            seen.add(key)
            unique_paths.append(resolved)
    return unique_paths


def _candidate_dll_dirs():
    paths = []
    prefixes = [Path(sys.base_prefix), Path(sys.exec_prefix), Path(sys.executable).resolve().parent]
    for prefix in prefixes:
        library_bin = prefix / 'Library' / 'bin'
        if library_bin.exists():
            paths.append(library_bin)

    for site_dir in site.getsitepackages():
        torch_lib = Path(site_dir) / 'torch' / 'lib'
        if torch_lib.exists():
            paths.append(torch_lib)

    user_site = site.getusersitepackages()
    if user_site:
        torch_lib = Path(user_site) / 'torch' / 'lib'
        if torch_lib.exists():
            paths.append(torch_lib)

    return _dedupe_paths(paths)


def prepare_torch_runtime():
    if os.name != 'nt':
        return []

    dll_dirs = _candidate_dll_dirs()
    if not dll_dirs:
        return []

    current_entries = _dedupe_paths([entry for entry in os.environ.get('PATH', '').split(os.pathsep) if entry])
    merged_entries = _dedupe_paths(dll_dirs + current_entries)
    os.environ['PATH'] = os.pathsep.join(str(path) for path in merged_entries)

    add_dll_directory = getattr(os, 'add_dll_directory', None)
    if add_dll_directory is not None:
        for dll_dir in dll_dirs:
            add_dll_directory(str(dll_dir))

    preload_order = [
        'sycl8.dll',
        'tbb12.dll',
        'mkl_core.2.dll',
        'libiomp5md.dll',
        'svml_dispmd.dll',
        'umf.dll',
        'ur_adapter_level_zero.dll',
        'ur_adapter_opencl.dll',
    ]
    for dll_name in preload_order:
        for dll_dir in dll_dirs:
            dll_path = dll_dir / dll_name
            if dll_path.exists():
                try:
                    ctypes.WinDLL(str(dll_path))
                except OSError:
                    pass
                break

    return [str(path) for path in dll_dirs]


def _xpu_available(torch_module):
    if not hasattr(torch_module, 'xpu'):
        return False
    try:
        return torch_module.xpu.is_available() and torch_module.xpu.device_count() > 0
    except Exception:
        return False


def _device_count(torch_module, device_type):
    if device_type == 'cuda':
        return torch_module.cuda.device_count()
    if device_type == 'xpu' and hasattr(torch_module, 'xpu'):
        return torch_module.xpu.device_count()
    return 0


def _parse_cuda_device_ids(devices, device_count):
    raw_value = str(devices).replace(' ', '')
    if not raw_value:
        raise ValueError('启用 CUDA 多卡时，--devices 不能为空。')

    parsed_ids = []
    seen = set()
    for part in raw_value.split(','):
        if not part:
            continue
        try:
            device_id = int(part)
        except ValueError as exc:
            raise ValueError(f'无法解析 CUDA 设备编号: {part!r}') from exc
        if device_id < 0 or device_id >= device_count:
            raise ValueError(
                f'CUDA 设备编号超出范围: {device_id}，当前可用设备数为 {device_count}。'
            )
        if device_id not in seen:
            parsed_ids.append(device_id)
            seen.add(device_id)

    if not parsed_ids:
        raise ValueError('启用 CUDA 多卡时，--devices 至少需要包含一个有效设备编号。')

    return parsed_ids


def resolve_device_config(args):
    import torch

    requested_accelerator = bool(getattr(args, 'use_gpu', True))
    device_type = 'cpu'
    if requested_accelerator and torch.cuda.is_available():
        device_type = 'cuda'
    elif requested_accelerator and _xpu_available(torch):
        device_type = 'xpu'

    requested_index = int(getattr(args, 'gpu', 0))
    device_count = _device_count(torch, device_type)
    if device_type != 'cpu' and device_count and requested_index >= device_count:
        requested_index = 0

    args.device_type = device_type
    args.gpu = requested_index
    args.device = 'cpu' if device_type == 'cpu' else f'{device_type}:{requested_index}'
    args.use_gpu = device_type != 'cpu'

    if device_type != 'cuda':
        args.use_multi_gpu = False
        args.device_ids = [requested_index] if device_type != 'cpu' else []
    else:
        if getattr(args, 'use_multi_gpu', False):
            args.device_ids = _parse_cuda_device_ids(getattr(args, 'devices', ''), device_count)
        else:
            args.device_ids = [requested_index]
        args.gpu = args.device_ids[0]
        args.device = f'cuda:{args.gpu}'

    if getattr(args, 'use_amp', False) and device_type != 'cuda':
        args.use_amp = False
        # print(f'[设备] 当前设备类型为 {device_type}，已自动关闭 AMP。')

    return args


def acquire_device(args):
    import torch

    device_type = getattr(args, 'device_type', 'cpu')
    if device_type == 'cuda':
        device = torch.device(getattr(args, 'device', f'cuda:{args.gpu}'))
        print(f'[设备] 使用 CUDA: {device}')
        return device

    if device_type == 'xpu':
        device = torch.device(getattr(args, 'device', f'xpu:{args.gpu}'))
        print(f'[设备] 使用 XPU: {device}')
        return device

    device = torch.device('cpu')
    print('[设备] 使用 CPU')
    return device


def autocast_context(args):
    import torch

    if getattr(args, 'use_amp', False) and getattr(args, 'device_type', 'cpu') == 'cuda':
        return torch.cuda.amp.autocast()
    return nullcontext()


def create_grad_scaler(args):
    import torch

    if getattr(args, 'use_amp', False) and getattr(args, 'device_type', 'cpu') == 'cuda':
        return torch.cuda.amp.GradScaler()
    return None


def _device_namespace(torch_module, device_type):
    if device_type == 'cuda' and hasattr(torch_module, 'cuda'):
        return torch_module.cuda
    if device_type == 'xpu' and hasattr(torch_module, 'xpu'):
        return torch_module.xpu
    return None


def synchronize_device(device_type):
    import torch

    namespace = _device_namespace(torch, device_type)
    if namespace is None:
        return
    synchronize = getattr(namespace, 'synchronize', None)
    if synchronize is None:
        return
    try:
        synchronize()
    except Exception:
        pass


def empty_cache(device_type):
    import torch

    if device_type == 'cuda' and hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()
    elif device_type == 'xpu' and hasattr(torch, 'xpu'):
        torch.xpu.empty_cache()


def reset_peak_memory_stats(device_type):
    import torch

    namespace = _device_namespace(torch, device_type)
    if namespace is None:
        return
    reset_fn = getattr(namespace, 'reset_peak_memory_stats', None)
    if reset_fn is None:
        return
    try:
        reset_fn()
    except Exception:
        pass


def device_memory_snapshot(device_type):
    import torch

    namespace = _device_namespace(torch, device_type)
    if namespace is None:
        return {}

    snapshot = {}
    metric_map = {
        'allocated': 'memory_allocated',
        'reserved': 'memory_reserved',
        'max_allocated': 'max_memory_allocated',
        'max_reserved': 'max_memory_reserved',
    }
    for metric_name, attr_name in metric_map.items():
        metric_fn = getattr(namespace, attr_name, None)
        if metric_fn is None:
            continue
        try:
            snapshot[metric_name] = int(metric_fn())
        except Exception:
            continue
    return snapshot



def manage_device_memory(device_type, force_gc=False, reset_peak=False):
    if force_gc:
        gc.collect()
    synchronize_device(device_type)
    empty_cache(device_type)
    if reset_peak:
        reset_peak_memory_stats(device_type)

def _normalize_model_state_dict(state_dict):
    """
    多卡训练时，将模型状态字典的键值前缀 'module.' 移除
    """
    if not isinstance(state_dict, dict):
        return state_dict

    if not any(isinstance(key, str) and key.startswith('module.') for key in state_dict.keys()):
        return state_dict

    return {
        key.removeprefix('module.'): value
        for key, value in state_dict.items()
    }


def load_training_checkpoint(path, device):
    import torch
    ## 兼容两种加载方式，如果 checkpoint 是一个包含 'model_state_dict' 键的字典，则直接返回；否则将整个 checkpoint 视为模型状态字典并进行规范化处理后返回，包含 'model_state_dict' 键的字典。
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
     # 直接保存的 model.state_dict() 会进入这里
    # 此时 checkpoint 是 OrderedDict，不是 dict
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        return {
            'model_state_dict': _normalize_model_state_dict(checkpoint),
        }

    # 完整 checkpoint（包含 epoch, loss 等）是 dict
    # 但内部的 model_state_dict 仍是 OrderedDict
    normalized_checkpoint['model_state_dict'] = _normalize_model_state_dict(checkpoint['model_state_dict'])
    return normalized_checkpoint