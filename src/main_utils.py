import yaml
import argparse


class Config:
    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                value = Config(value)  # recursively convert nested dicts
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out

    def to_yaml(self) -> str:
        """
        Return a clean YAML string with no Python-specific annotations.
        """
        plain = self.to_dict()
        return yaml.safe_dump(
            plain,
            sort_keys=False,  # preserve insertion order
            default_flow_style=False,  # block style YAML
            allow_unicode=True,
        )

    def save_yaml(self, path: str):
        """
        Save config as clean YAML.
        """
        with open(path, "w") as f:
            f.write(self.to_yaml())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug_scipy.yaml",
        help="Path to config YAML",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = Config(cfg)
    return cfg


def load_registry():
    """
    Import all modules that register objects into the registry.

    This must be called exactly once at startup (e.g. in main.py).
    """
    # These imports trigger decorators like @registry.register_trainer(...)
    import src.trainers  # NOQA
    import src.models  # NOQA
    import src.losses  # NOQA
    import src.datasets  # NOQA
    import src.ed_models  # NOQA
