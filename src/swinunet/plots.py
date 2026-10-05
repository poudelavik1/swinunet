'Plots for hairline crack segmentation.'

from __future__ import annotations
from .common import BASIC_METRICS, COUNT_NAMES, METRIC_NAMES, Path, csv, json, np
from .evaluation import confusion_from_counts, metrics_from_counts

MODE_LABELS = {"no_tta": "no TTA", "tta": "flip TTA"}
PLOT_INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
# Fixed colour order (blue, orange, aqua): distinguishable with colour-vision deficiency.
SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
BLUE_RAMP = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")  # one hue, light -> dark
PLOT_STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.edgecolor": PLOT_INK["axis"], "axes.labelcolor": PLOT_INK["secondary"],
    "axes.titlecolor": PLOT_INK["primary"], "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": PLOT_INK["grid"], "grid.linewidth": 0.8,
    "grid.linestyle": "-", "xtick.color": PLOT_INK["muted"], "ytick.color": PLOT_INK["muted"],
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "legend.frameon": False,
    "text.color": PLOT_INK["primary"],
}


def read_history(path: Path) -> dict[str, np.ndarray]:
    """training_history.csv as {column: values}; empty when the file is missing."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    history = {}
    for name in (rows[0] if rows else ()):
        try:
            history[name] = np.array([float(row[name]) for row in rows])
        except (TypeError, ValueError):
            continue
    return history


def save_evaluation_tables(folder: Path, thresholds, evaluation: dict) -> None:
    """The numbers behind the figures: evaluation_curves.csv and confusion_matrices.json."""
    folder.mkdir(parents=True, exist_ok=True)
    curve_names = (*METRIC_NAMES, "false_positive_rate")
    confusion = {}
    with (folder / "evaluation_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "split", "threshold", *COUNT_NAMES, "tn", *curve_names))
        for mode, data in evaluation.items():
            index = data["threshold_index"]
            confusion[mode] = {"threshold": float(thresholds[index])}
            for split, name in (("val", "validation"), ("test", "test")):
                counts, metrics = data[split], metrics_from_counts(data[split])
                for row, threshold in enumerate(thresholds):
                    matrix = confusion_from_counts(counts[row])
                    writer.writerow((mode, name, f"{threshold:.3f}", *(int(round(value)) for value in counts[row]),
                                     matrix["tn"], *(f"{metrics[key][row]:.6f}" for key in curve_names)))
                matrix = confusion_from_counts(counts[index])
                confusion[mode][name] = {
                    **matrix, **{key: float(metrics[key][index]) for key in curve_names},
                    "specificity": matrix["tn"] / max(1, matrix["tn"] + matrix["fp"]),
                }
    (folder / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2))


def save_plots(output: Path, thresholds, evaluation: dict, best_epoch: int, tolerance: int) -> list[str]:
    """Write every figure to <output>/plots and return the file names.

    `evaluation` maps "no_tta"/"tta" to {"threshold_index", "val", "test"} with
    count arrays from ThresholdSweep.count_array(). Epoch curves come from
    training_history.csv; figures needing columns an older history lacks are skipped.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    folder = output / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    ink, written = PLOT_INK, []
    blue, orange, aqua = SERIES_COLOURS
    thresholds = np.asarray(thresholds, dtype=float)

    def finish(figure, name, epoch_axis=False):
        if epoch_axis:  # epochs are whole numbers
            for axis in figure.axes:
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        figure.savefig(folder / name)
        plt.close(figure)
        written.append(name)

    def note(axis, text, offset=-0.22):
        axis.text(0, offset, text, transform=axis.transAxes, fontsize=8, color=ink["muted"], va="top")

    def mark(axis, x, text):
        axis.axvline(x, color=ink["muted"], linewidth=1.0, zorder=1)
        axis.annotate(text, (x, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -3),
                      textcoords="offset points", va="top", fontsize=8.5, color=ink["secondary"])

    def lines(axis, x, series, end_values=True, pad=0.10):
        """series = [(label, values, colour)]. End values are written only where they do
        not collide; the legend always carries the identity."""
        for label, values, colour in series:
            axis.plot(x, values, color=colour, label=label)
        span = (x[-1] - x[0]) or 1.0
        axis.set_xlim(x[0] - 0.01 * span, x[-1] + (pad if end_values else 0.01) * span)
        if end_values:
            low, high = axis.get_ylim()
            placed = []
            for _, values, _ in sorted(series, key=lambda item: -item[1][-1]):
                if all(abs(values[-1] - other) > 0.06 * (high - low) for other in placed):
                    axis.annotate(f"{values[-1]:.3f}", (x[-1], values[-1]), xytext=(5, 0),
                                  textcoords="offset points", va="center", fontsize=8.5, color=ink["secondary"])
                    placed.append(values[-1])

    def legend_in_title_row(axis, columns):
        """Single panel: legend right of the left-aligned title, clear of the data."""
        axis.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=columns, borderaxespad=0.2,
                    handlelength=1.6, columnspacing=1.4)

    def legend_in_heading_row(axes, source_axis, columns):
        """Several panels: one legend at the right of the figure heading."""
        axes[-1].legend(*source_axis.get_legend_handles_labels(), loc="lower right", bbox_to_anchor=(1.0, 1.09),
                        ncol=columns, borderaxespad=0, handlelength=1.6, columnspacing=1.4)

    def dot(axis, x, y, colour):
        axis.plot(x, y, "o", color=colour, markersize=8, markeredgecolor="white", markeredgewidth=2, zorder=4)

    with plt.rc_context(PLOT_STYLE):
        history = read_history(output / "training_history.csv")
        if "epoch" in history and "train_loss" in history:
            epochs = history["epoch"]
            show_best = bool(best_epoch) and epochs[0] <= best_epoch <= epochs[-1]

            figure, axis = plt.subplots(figsize=(7, 4.2))
            lines(axis, epochs, [("Training", history["train_loss"], blue), ("Validation", history["val_loss"], orange)])
            if show_best:
                mark(axis, best_epoch, f"best epoch {best_epoch}")
            axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
            legend_in_title_row(axis, 2)
            note(axis, "Training loss also contains the auxiliary deep-supervision terms and is measured on augmented "
                       "crops,\nso it sits above the validation loss (full target images, main output only).")
            finish(figure, "loss_curves.png", epoch_axis=True)

            if all(f"{split}_{name}" in history for split in ("train", "val") for name in BASIC_METRICS):
                titles = {"precision": "Precision", "recall": "Recall", "f1": "F1 score", "iou": "IoU",
                          "accuracy": "Pixel accuracy"}
                figure, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
                for axis, name in zip(axes.flat, BASIC_METRICS):
                    lines(axis, epochs, [("Training", history[f"train_{name}"], blue),
                                         ("Validation", history[f"val_{name}"], orange)], pad=0.17)
                    if show_best:
                        mark(axis, best_epoch, "best")
                    axis.set_title(titles[name])
                for axis in (axes[1, 0], axes[1, 1], axes[0, 2]):
                    axis.set_xlabel("Epoch")
                axes[0, 2].tick_params(labelbottom=True)
                axes[1, 2].axis("off")
                axes[1, 2].legend(*axes[0, 0].get_legend_handles_labels(), loc="upper left",
                                  title="Both at threshold 0.5")
                note(axes[1, 2], "Training: augmented crops, all sources.\nValidation: full target images, EMA weights.\n"
                                 "Pixel accuracy is dominated by background\nand stays near 1; judge cracks by F1 and IoU.",
                     offset=0.62)
                figure.suptitle("Training and validation metrics per epoch", x=0.07, ha="left",
                                fontsize=12, fontweight="semibold")
                finish(figure, "metric_curves_train_val.png", epoch_axis=True)

            if all(name in history for name in ("precision", "recall", "f1", "iou", "tol_f1")):
                figure, axes = plt.subplots(1, 3, figsize=(14, 4))
                for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                            (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                    lines(axis, epochs, [("Precision", history[prefix + "precision"], blue),
                                         ("Recall", history[prefix + "recall"], orange),
                                         ("F1", history[prefix + "f1"], aqua)], pad=0.14)
                    axis.set(xlabel="Epoch", title=title)
                # One series: the title names it, and a neutral ink keeps it from reading as "Precision".
                lines(axes[2], epochs, [("IoU", history["iou"], ink["secondary"])], pad=0.14)
                axes[2].set(xlabel="Epoch", title="IoU")
                legend_in_heading_row(axes, axes[0], 3)
                if show_best:
                    for axis in axes:
                        mark(axis, best_epoch, f"best epoch {best_epoch}")
                figure.suptitle("Validation metrics per epoch (target images, at each epoch's best threshold)",
                                x=0.07, ha="left", fontsize=12, fontweight="semibold")
                finish(figure, "validation_metric_curves.png", epoch_axis=True)

            if "lr" in history:
                figure, axis = plt.subplots(figsize=(7, 3.6))
                lines(axis, epochs, [("Learning rate", history["lr"], blue)], end_values=False)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-4, -4))
                axis.set(xlabel="Epoch", ylabel="Decoder learning rate", title="Learning-rate schedule (warm-up, cosine decay)")
                finish(figure, "learning_rate.png", epoch_axis=True)

        coverage_path = output / "contrast_coverage.json"
        if coverage_path.is_file():  # written by training in the "all" contrast mode
            coverage = json.loads(coverage_path.read_text())
            changes, draws = coverage["contrast_change_percent"], coverage["draws_per_level"]
            spacing = min(np.diff(changes)) if len(changes) > 1 else 1.0
            figure, axis = plt.subplots(figsize=(7.5, 3.8))
            axis.bar(changes, draws, width=0.6 * spacing, color=blue)
            axis.grid(axis="x", visible=False)
            ticks = changes[::2] if len(changes) > 12 else changes
            axis.set_xticks(ticks, [f"{value:+g}" if value else "0" for value in ticks])
            axis.set(xlabel="Contrast change (%)", ylabel="Training samples",
                     title=f"Training samples at each of the {coverage['levels']} contrast levels")
            note(axis, f"{coverage['images_trained_at_every_level']:,} of {coverage['training_images']:,} training "
                       f"images were trained at every level (mean {coverage['mean_draws_per_image']:.1f} draws per "
                       "image).", offset=-0.24)
            finish(figure, "contrast_level_coverage.png")

        colour_map = LinearSegmentedColormap.from_list("crack_blue", BLUE_RAMP)
        for mode, data in evaluation.items():
            label, index = MODE_LABELS.get(mode, mode), data["threshold_index"]
            selected = thresholds[index]
            splits = (("Validation", metrics_from_counts(data["val"]), orange),
                      ("Test", metrics_from_counts(data["test"]), aqua))
            validation = splits[0][1]

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
            for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                        (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                lines(axis, thresholds, [("Precision", validation[prefix + "precision"], blue),
                                         ("Recall", validation[prefix + "recall"], orange),
                                         ("F1", validation[prefix + "f1"], aqua)], end_values=False)
                mark(axis, selected, f"selected {selected:.3f}")
                axis.set(xlabel="Probability threshold", title=title, ylim=(0, 1))
            axes[0].set_ylabel("Score")
            legend_in_heading_row(axes, axes[0], 3)
            figure.suptitle(f"Validation threshold sweep ({label})", x=0.07, ha="left", fontsize=12,
                            fontweight="semibold")
            finish(figure, f"threshold_sweep_{mode}.png")

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
            for axis, prefix, title in ((axes[0], "", "Strict pixel metrics"),
                                        (axes[1], "tol_", f"{tolerance}-px tolerance metrics")):
                for name, metrics, colour in splits:
                    axis.plot(metrics[prefix + "recall"], metrics[prefix + "precision"], color=colour, label=name)
                    dot(axis, metrics[prefix + "recall"][index], metrics[prefix + "precision"][index], colour)
                axis.set(xlabel="Recall", title=title, xlim=(0, 1), ylim=(0, 1))
            axes[0].set_ylabel("Precision")
            legend_in_heading_row(axes, axes[0], 2)
            note(axes[0], f"Curves cover thresholds {thresholds[0]:.2f}-{thresholds[-1]:.2f}; "
                          f"dots mark the selected threshold {selected:.3f}.", offset=-0.2)
            figure.suptitle(f"Precision-recall curves ({label})", x=0.07, ha="left", fontsize=12,
                            fontweight="semibold")
            finish(figure, f"precision_recall_curve_{mode}.png")

            figure, axis = plt.subplots(figsize=(6.4, 4.4))
            for name, metrics, colour in splits:
                axis.plot(metrics["false_positive_rate"], metrics["recall"], color=colour, label=name)
                dot(axis, metrics["false_positive_rate"][index], metrics["recall"][index], colour)
            axis.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
                     title=f"ROC curves ({label})", xlim=(0, None), ylim=(0, 1))
            legend_in_title_row(axis, 2)
            background = 100 * (1 - data["test"][0, 6] / max(1.0, data["test"][0, 7]))
            note(axis, f"Thresholds {thresholds[0]:.2f}-{thresholds[-1]:.2f}; dots mark the selected threshold "
                       f"{selected:.3f}.\nBackground is {background:.1f} % of the test pixels, so the false "
                       "positive rate stays small.", offset=-0.19)
            finish(figure, f"roc_curve_{mode}.png")

            figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
            for axis, name, counts in ((axes[0], "Validation", data["val"]), (axes[1], "Test", data["test"])):
                cells = confusion_from_counts(counts[index])
                matrix = np.array([[cells["tp"], cells["fn"]], [cells["fp"], cells["tn"]]], dtype=float)
                share = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
                image = axis.imshow(share, cmap=colour_map, vmin=0, vmax=1)
                for row in range(2):
                    for column in range(2):
                        axis.text(column, row, f"{int(matrix[row, column]):,}\n{100 * share[row, column]:.2f} %",
                                  ha="center", va="center", fontsize=11,
                                  color="white" if share[row, column] > 0.5 else ink["primary"])
                axis.set_xticks([0, 1], ["Crack", "Background"])
                axis.set_yticks([0, 1], ["Crack", "Background"], rotation=90, va="center")
                axis.set(xlabel="Predicted", ylabel="Actual", title=f"{name} (threshold {selected:.3f})")
                axis.grid(False)
                axis.set_xticks([0.5], minor=True)
                axis.set_yticks([0.5], minor=True)
                axis.grid(which="minor", color="white", linewidth=2)  # surface gap between cells
                axis.tick_params(which="both", length=0, labelcolor=ink["secondary"])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
            bar.set_label("Share of the actual class (row)", color=ink["secondary"])
            bar.outline.set_visible(False)
            figure.suptitle(f"Pixel confusion matrices ({label}): count and share of each actual class",
                            x=0.07, ha="left", fontsize=12, fontweight="semibold")
            finish(figure, f"confusion_matrix_{mode}.png")
    return written


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

__all__ = ['MODE_LABELS', 'PLOT_INK', 'SERIES_COLOURS', 'BLUE_RAMP', 'PLOT_STYLE', 'read_history', 'save_evaluation_tables', 'save_plots']
