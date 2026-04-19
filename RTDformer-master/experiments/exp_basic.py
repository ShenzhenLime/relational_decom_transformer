import importlib
import torch
from utils.device_utils import acquire_device


MODEL_MODULES = {
    'FourierGNN': 'model.FourierGNN',
    'Transformer': 'model.Transformer',
    'TDformer': 'model.TDformer',
    'Informer': 'model.Informer',
    'Wformer': 'model.Wformer',
    'iTransformer': 'model.iTransformer',
    'RTDformer2': 'model.RTDformer2',
    'FEDformer': 'model.FEDformer',
    'PDF': 'model.PDF',
    'StockMixer': 'model.StockMixer',
    'DLinear': 'model.DLinear',
}


def load_model_module(model_name):
    module_name = MODEL_MODULES.get(model_name)
    if module_name is None:
        raise KeyError(f'未知模型: {model_name}')
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'无法导入模型 {model_name} 对应模块 {module_name}。'
            '如果你当前运行的是 RTDformer2 云端最小部署包，请只使用 --model RTDformer2，'
            '因为该部署包不会包含其他模型文件。'
        ) from exc

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            args.model: load_model_module(args.model)
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
    ## 父类直接依赖的方法，通常要在父类先定义接口，比如 _build_model、test、train是子类必须实现这些方法
    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        return acquire_device(self.args)

    def _get_data(self):
        raise NotImplementedError("Subclasses must implement _get_data()")

    def vali(self):
        raise NotImplementedError("Subclasses must implement vali()")

    def train(self):
        raise NotImplementedError("Subclasses must implement train()")

    def test(self):
        raise NotImplementedError("Subclasses must implement test()")
