from src.util import Config

_MODEL_REGISTRY = {}

def register_model(name):
    def decorator(fn):
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator

def build_model(cfg: Config):
    try:
        builder = _MODEL_REGISTRY[cfg.name]
    except KeyError:
        raise ValueError(f"Unknown model name: {cfg.name}") from None
    return builder(cfg)