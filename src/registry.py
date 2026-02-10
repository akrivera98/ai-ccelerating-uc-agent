class Registry:
    def __init__(self):
        self._trainers = {}
        self._models = {}
        self._losses = {}
        self._datasets = {}
        self._ed_models = {}

    # Register methods

    def register_trainer(self, name: str):
        def decorator(cls):
            if name in self._trainers:
                raise KeyError(f"Trainer '{name}' already registered.")
            self._trainers[name] = cls
            return cls

        return decorator

    def register_model(self, name: str):
        def decorator(cls):
            if name in self._models:
                raise KeyError(f"Model '{name}' already registered.")
            self._models[name] = cls
            return cls

        return decorator

    def register_loss(self, name: str):
        def decorator(cls):
            if name in self._losses:
                raise KeyError(f"Loss '{name}' already registered.")
            self._losses[name] = cls
            return cls

        return decorator

    def register_dataset(self, name: str):
        def decorator(cls):
            if name in self._datasets:
                raise KeyError(f"Dataset '{name}' already registered.")
            self._datasets[name] = cls
            return cls

        return decorator

    def register_ed_model(self, name: str):
        def decorator(cls):
            if name in self._ed_models:
                raise KeyError(f"ED model '{name}' already registered.")
            self._ed_models[name] = cls
            return cls

        return decorator

    # Get methods

    def get_trainer(self, name: str):
        return self._get(name, self._trainers, "Trainer")

    def get_model(self, name: str):
        return self._get(name, self._models, "Model")

    def get_loss(self, name: str):
        return self._get(name, self._losses, "Loss")

    def get_dataset(self, name: str):
        return self._get(name, self._datasets, "Dataset")

    def get_ed_model(self, name: str):
        return self._get(name, self._ed_models, "ED model")

    # Internal helper

    @staticmethod
    def _get(name, table, kind):
        if name not in table:
            available = ", ".join(sorted(table.keys()))
            raise KeyError(f"{kind} '{name}' not found. Available: [{available}]")
        return table[name]


registry = Registry()
