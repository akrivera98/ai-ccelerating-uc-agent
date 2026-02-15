from src.registry import registry
from src.utils.main_utils import parse_args, load_config, load_registry


def main():
    args = parse_args()
    cfg = load_config(args.config, args)

    load_registry()

    # Instantiate trainer (trainer builds model, data, loss, optimizer)
    trainer_name = cfg.trainer.name

    trainer = registry.get_trainer(trainer_name)(
        cfg,
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
