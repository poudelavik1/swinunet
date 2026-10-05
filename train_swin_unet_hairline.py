"""Compatibility entry point for the packaged Swin U-Net trainer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from swinunet.models import load_model_from_checkpoint
from swinunet.training import main

if __name__ == "__main__":
    main()
