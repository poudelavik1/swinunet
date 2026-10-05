"""Print completed epoch metrics from training_history.csv without GPU dependencies."""
import argparse
import csv
import time
from pathlib import Path


def read_rows(path):
    if not path.is_file():
        return []
    # Ignore an incomplete trailing line while the trainer is writing it.
    text = path.read_text(encoding="utf-8")
    text = text[:text.rfind("\n") + 1]
    return list(csv.DictReader(text.splitlines()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Run output directory")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    path = args.output / "training_history.csv"
    print("Epoch  Train loss  Val loss   Val P    Val R    Val F1   Val IoU  Best-thr tolF1  Seconds", flush=True)
    shown = set()
    waiting = False
    try:
        while True:
            rows = read_rows(path)
            if not rows and not waiting:
                print(f"Waiting for the first completed epoch: {path}", flush=True)
                waiting = True
            for row in rows:
                try:
                    epoch = int(row["epoch"])
                    values = [float(row[key]) for key in (
                        "train_loss", "val_loss", "val_precision", "val_recall",
                        "val_f1", "val_iou", "tol_f1", "seconds")]
                except (KeyError, TypeError, ValueError):
                    continue
                if epoch not in shown:
                    print(f"{epoch:5d}  " + "  ".join(f"{v:8.4f}" for v in values[:-1]) + f"  {values[-1]:7.0f}", flush=True)
                    shown.add(epoch)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
