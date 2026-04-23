import pickle
import importlib
import sys

import joblib

"""
Custom unpickler qui redirige les classes ML vers leurs vrais modules
peu importe depuis quel contexte le pickle est chargé.
"""
_CLASS_MAP = {
    'TabNetEnsembleWrapper': ('src.ml.tabnet_utils', 'TabNetEnsembleWrapper'),
    'RaceContextEncoder': ('src.ml.features', 'RaceContextEncoder'),
    'PmuFeatureEngineer': ('src.ml.features', 'PmuFeatureEngineer'),
    'HyperStackModel': ('src.ml.models', 'HyperStackModel'),  # ← models.py
    'TabNetBridge':          ('src.ml.tabnet_bridge', 'TabNetBridge'),  # nouveau
    'LTRRankerWrapper': ('src.ml.trainer_ltr', 'LTRRankerWrapper'),
    'GPTModelWrapper': ('src.ml.trainer_gpt', 'GPTModelWrapper'),
    'PMUTransformer': ('src.ml.trainer_gpt', 'PMUTransformer'),  # nouveau
}

def _patch_main():
    """Injecte toutes les classes ML dans __main__ pour que joblib.load les trouve."""
    main = sys.modules.get('__main__')
    if main is None:
        return
    for class_name, (module_path, attr) in _CLASS_MAP.items():
        if not hasattr(main, class_name):
            try:
                mod = importlib.import_module(module_path)
                setattr(main, class_name, getattr(mod, attr))
            except Exception:
                pass

def safe_load(path):
    """Charge un pipeline ML joblib en résolvant les classes __main__."""
    _patch_main()
    return joblib.load(path)
