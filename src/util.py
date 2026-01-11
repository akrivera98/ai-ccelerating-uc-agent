import yaml

class Config:
    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                value = Config(value)  # recursively convert nested dicts
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"
    
def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        raw_cfg = f.read().replace("!!python/object:src.util.Config", "")
        cfg = yaml.safe_load(raw_cfg)
    return Config(cfg)