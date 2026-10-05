"""Training entry point with TOML defaults and command-line overrides."""
from pathlib import Path
import argparse
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def configured_arguments(config, arguments):
    settings = tomllib.loads(config.read_text(encoding="utf-8"))["training"]
    defaults = []
    for key, value in settings.items():
        if key not in {"epochs", "batch_size", "workers"}:
            raise ValueError(f"Unsupported config key: {key}")
        if type(value) is not int or value < (0 if key == "workers" else 1):
            raise ValueError(f"Invalid {key}: {value}")
        defaults.extend(["--" + key.replace("_", "-"), str(value)])
    return defaults + arguments


if __name__ == "__main__":
    from swinunet.training import main
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/training.toml")
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *configured_arguments(options.config, remaining)]
    main()
