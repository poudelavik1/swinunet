"""Detect cracks in an image and draw them, in millimetres, over the image in AutoCAD.

1. Crack detection runs through ``hairline_crack_detection.py``. If its results
   already exist in the output folder they are reused (``--redetect`` runs it
   again). Detection options such as ``--threshold`` or ``--epsilon`` are
   passed straight through.
2. AutoCAD: a running session is detected and used. If none is running, the
   script asks whether to start one (``--start-autocad`` starts it without
   asking). Cracks are drawn in the active drawing, or in a new metric drawing
   with ``--new-drawing`` or when no drawing is open.
3. Drawing units are set to millimetres (INSUNITS = 4, decimal LUNITS,
   MEASUREMENT = 1).
4. Scale: ``--mm-per-pixel`` or ``--image-width-mm``. Without either, use the
   image pixel dimensions: 1 pixel = 1 drawing unit (nominally 1 mm).
   This default is uncalibrated; lengths are not physical measurements.
5. The image is inserted at the origin at the chosen scale and sent to the back.
   Every crack polyline is drawn above it on layer CRACKS, and each crack's
   length is written next to it on layer CRACK_LABELS, e.g.
   ``C12  L = 345.6 mm``. A summary note sits above the image.
6. ``crack_lengths_mm.csv`` and ``crack_polylines_mm.csv`` are written to the
   output folder.

Crack lengths are measured along the drawn polylines, so each label matches the
Length that AutoCAD reports for that crack's polylines.

``hairline_crack_detection.py`` calls this drawing step automatically when its
detection finishes (``--no-autocad`` there stops after detection), so either
script can be used; this one skips detection when results already exist.

    python cracks_to_autocad.py "type a front.png" --mm-per-pixel 0.30
    python cracks_to_autocad.py "type a front.png"          (use image pixel scale)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIRECTORY))

LAYERS = {  # name: (AutoCAD colour index, lineweight in 1/100 mm)
    "CRACK_IMAGE": (7, -3),
    "CRACKS": (1, 35),
    "CRACK_LABELS": (2, -3),
}
INSUNITS_NAMES = {0: "unitless", 1: "inches", 2: "feet", 4: "millimetres", 5: "centimetres", 6: "metres"}
BUSY_HRESULTS = (-2147418111, -2147417846)  # RPC_E_CALL_REJECTED, RPC_E_SERVERCALL_RETRYLATER


# --------------------------------------------------------------------------- #
# Crack geometry (independent of AutoCAD)
# --------------------------------------------------------------------------- #
def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        return image.size  # (width, height)


def read_polylines(folder: Path) -> tuple[dict[int, dict], dict[int, float]]:
    """RDP polylines {polyline_id: {"crack": id, "points": [(x_px, y_px)]}} and
    dense skeleton length per polyline in pixels."""
    polylines: dict[int, dict] = {}
    with (folder / "crack_polylines_rdp.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            entry = polylines.setdefault(int(row["polyline_id"]), {"crack": int(row["crack_id"]), "points": []})
            entry["points"].append((int(row["vertex_order"]), float(row["x_px"]), float(row["y_px"])))
    for entry in polylines.values():
        entry["points"] = [(x, y) for _, x, y in sorted(entry["points"])]
    skeleton_px = {}
    summary = folder / "crack_summary.csv"
    if summary.is_file():
        with summary.open(newline="", encoding="utf-8") as handle:
            skeleton_px = {int(row["polyline_id"]): float(row["length_px"]) for row in csv.DictReader(handle)}
    return polylines, skeleton_px


def to_mm(points_px, image_height: int, mm_per_pixel: float) -> list[tuple[float, float]]:
    """Pixel (column, row) -> drawing mm, pixel centres, origin at the image's bottom-left."""
    return [((x + 0.5) * mm_per_pixel, (image_height - y - 0.5) * mm_per_pixel) for x, y in points_px]


def path_length(points) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def point_at_half_length(points):
    """Point halfway along a polyline, used to place its label."""
    half, walked = path_length(points) / 2, 0.0
    for a, b in zip(points, points[1:]):
        step = math.dist(a, b)
        if walked + step >= half and step > 0:
            t = (half - walked) / step
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        walked += step
    return points[len(points) // 2]


def build_cracks(polylines, skeleton_px, image_height, mm_per_pixel):
    """Group polylines into cracks, measure them in mm and number them left to right."""
    grouped = defaultdict(list)
    for polyline_id, entry in polylines.items():
        points_mm = to_mm(entry["points"], image_height, mm_per_pixel)
        grouped[entry["crack"]].append({
            "id": polyline_id, "points": points_mm, "length": path_length(points_mm),
            "skeleton_length": skeleton_px.get(polyline_id, math.nan) * mm_per_pixel,
        })
    cracks = []
    for members in grouped.values():
        xs = [x for member in members for x, _ in member["points"]]
        ys = [y for member in members for _, y in member["points"]]
        longest = max(members, key=lambda member: member["length"])
        cracks.append({
            "polylines": members, "length": sum(m["length"] for m in members),
            "skeleton_length": sum(m["skeleton_length"] for m in members),
            "vertices": sum(len(m["points"]) for m in members),
            "bounds": (min(xs), min(ys), max(xs), max(ys)),
            "anchor": point_at_half_length(longest["points"]),
        })
    cracks.sort(key=lambda crack: ((crack["bounds"][0] + crack["bounds"][2]) / 2, -crack["bounds"][3]))
    for number, crack in enumerate(cracks, 1):
        crack["name"] = f"C{number}"
    return cracks


def write_csvs(folder: Path, cracks, mm_per_pixel: float) -> None:
    with (folder / "crack_lengths_mm.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("crack", "length_mm", "skeleton_length_mm", "polylines", "vertices",
                         "min_x_mm", "min_y_mm", "max_x_mm", "max_y_mm", "label_x_mm", "label_y_mm",
                         "mm_per_pixel"))
        for crack in cracks:
            writer.writerow((crack["name"], f"{crack['length']:.2f}", f"{crack['skeleton_length']:.2f}",
                             len(crack["polylines"]), crack["vertices"],
                             *(f"{value:.2f}" for value in crack["bounds"]),
                             *(f"{value:.2f}" for value in crack["anchor"]), f"{mm_per_pixel:.6f}"))
    with (folder / "crack_polylines_mm.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("polyline_id", "crack", "vertex_order", "x", "y"))
        for crack in cracks:
            for polyline in crack["polylines"]:
                for order, (x, y) in enumerate(polyline["points"], 1):
                    writer.writerow((polyline["id"], crack["name"], order, f"{x:.3f}", f"{y:.3f}"))


# --------------------------------------------------------------------------- #
# AutoCAD (COM)
# --------------------------------------------------------------------------- #
def com(call, attempts: int = 240, delay: float = 0.5):
    """Run one COM call, retrying for up to 2 minutes while AutoCAD is busy
    (starting, loading a large image, regenerating).

    A busy AutoCAD rejects calls. pywin32 normally reports that as a com_error,
    but the first time a method or property name is used its dynamic lookup
    swallows the rejection and raises AttributeError instead (e.g.
    "AutoCAD.Application.ZoomExtents"), so AttributeError is retried as well.
    """
    for attempt in range(attempts):
        try:
            return call()
        except AttributeError:
            if attempt < attempts - 1:
                time.sleep(delay)
                continue
            raise
        except Exception as exc:
            code = getattr(exc, "hresult", None) or (exc.args[0] if exc.args else None)
            if code in BUSY_HRESULTS and attempt < attempts - 1:
                time.sleep(delay)
                continue
            raise


def wait_until_idle(application, timeout: float = 300.0) -> None:
    """Block until AutoCAD is quiescent (not starting, loading an image or regenerating)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if application.GetAcadState().IsQuiescent:
                return
        except Exception:
            pass  # rejected while busy
        time.sleep(0.5)


def ask_yes_no(title: str, question: str) -> bool:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        answer = messagebox.askyesno(title, question, parent=root)
        root.destroy()
        return bool(answer)
    except Exception:
        try:
            return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
        except EOFError:
            return False


def connect_autocad(start: bool):
    """Return (application, started_by_us). Detects a running AutoCAD first."""
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise SystemExit("AutoCAD control needs pywin32: py -m pip install pywin32") from exc
    pythoncom.CoInitialize()
    try:
        application = win32com.client.GetActiveObject("AutoCAD.Application")
        print(f"AutoCAD is running: {com(lambda: application.Name)} {com(lambda: application.Version)}, "
              f"{com(lambda: application.Documents.Count)} drawing(s) open.")
        return application, False
    except Exception:
        print("AutoCAD is not running.")
    if not start and not ask_yes_no("AutoCAD", "AutoCAD is not running.\n\nStart AutoCAD now?"):
        raise SystemExit("Open AutoCAD and run this script again (or pass --start-autocad).")
    print("Starting AutoCAD; this can take a minute ...")
    application = win32com.client.Dispatch("AutoCAD.Application")
    com(lambda: setattr(application, "Visible", True))
    wait_until_idle(application)
    print(f"AutoCAD started: {com(lambda: application.Version)}")
    return application, True


def open_drawing(application, new_drawing: bool):
    """Return (document, created_by_us)."""
    if not new_drawing and com(lambda: application.Documents.Count) > 0:
        document = com(lambda: application.ActiveDocument)
        print(f"Drawing in the active drawing: {com(lambda: document.Name)}")
        return document, False
    for template in ("acadiso.dwt", None):  # metric template first
        try:
            document = com(lambda: application.Documents.Add(template) if template else application.Documents.Add())
            print(f"Created a new drawing{' from acadiso.dwt' if template else ''}: {com(lambda: document.Name)}")
            return document, True
        except Exception:
            continue
    raise SystemExit("AutoCAD could not create a new drawing.")


def set_millimetres(document) -> None:
    import pythoncom
    from win32com.client import VARIANT

    previous = int(com(lambda: document.GetVariable("INSUNITS")))
    for name, value in (("INSUNITS", 4), ("LUNITS", 2), ("LUPREC", 2), ("MEASUREMENT", 1)):
        com(lambda: document.SetVariable(name, VARIANT(pythoncom.VT_I2, value)))
    if previous == 4:
        print("Drawing units: millimetres (unchanged).")
    else:
        print(f"Drawing units changed from {INSUNITS_NAMES.get(previous, previous)} to millimetres.")
        if com(lambda: document.ModelSpace.Count) > 0:
            print("  Note: existing objects in this drawing were not rescaled; "
                  "use --new-drawing for a clean metric drawing.")
    if int(com(lambda: document.GetVariable("INSUNITS"))) != 4:
        raise SystemExit("AutoCAD refused to switch the drawing units to millimetres.")


def ensure_layer(document, name: str, colour: int, lineweight: int):
    try:
        layer = com(lambda: document.Layers.Item(name))
    except Exception:
        layer = com(lambda: document.Layers.Add(name))
    for attribute, value in (("Color", colour), ("LayerOn", True), ("Freeze", False), ("Lock", False),
                             ("Lineweight", lineweight)):
        try:
            com(lambda: setattr(layer, attribute, value))
        except Exception:
            pass  # e.g. cannot freeze/thaw the current layer; harmless
    return layer


def point(x: float, y: float):
    import pythoncom
    from win32com.client import VARIANT

    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [float(x), float(y), 0.0])


def insert_image(document, image_path: Path, width_px: int, height_px: int, mm_per_pixel: float):
    """Insert the image with its lower-left corner at the origin, one pixel = mm_per_pixel mm."""
    model_space = com(lambda: document.ModelSpace)
    raster = com(lambda: model_space.AddRaster(str(image_path.resolve()), point(0, 0), 1.0, 0.0))
    com(lambda: setattr(raster, "Layer", "CRACK_IMAGE"))
    try:
        com(lambda: setattr(raster, "ImageWidth", width_px * mm_per_pixel))
        com(lambda: setattr(raster, "ImageHeight", height_px * mm_per_pixel))
    except Exception:  # some releases keep the aspect ratio and expose one writable side
        com(lambda: setattr(raster, "ImageWidth", width_px * mm_per_pixel))
    try:
        com(lambda: setattr(raster, "Transparency", True))  # transparent PNG background
    except Exception:
        pass
    com(lambda: raster.Update())
    return raster


def calibrate(application, document, raster) -> float:
    """Pick two points on the image (drawn at 1 unit per pixel) and type their real distance."""
    import pythoncom
    from win32com.client import VARIANT

    wait_until_idle(application)
    com(lambda: application.ZoomExtents())
    wait_until_idle(application)
    try:
        import win32gui

        win32gui.SetForegroundWindow(com(lambda: application.HWND))
    except Exception:
        pass
    print("\nScale calibration: switch to AutoCAD, pick two points of a known distance on the image\n"
          "(for example both ends of the specimen), then type that distance in mm.")
    utility = com(lambda: document.Utility)
    first = com(lambda: utility.GetPoint(pythoncom.Missing, "\nFirst point of a known distance on the image: "))
    second = com(lambda: utility.GetPoint(VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, list(first)),
                                          "\nSecond point: "))
    distance_px = math.dist(first[:2], second[:2])
    if distance_px <= 0:
        raise SystemExit("The two points are identical; calibration cancelled.")
    distance_mm = float(com(lambda: utility.GetReal(f"\nReal distance for these {distance_px:.1f} px in mm: ")))
    if distance_mm <= 0:
        raise SystemExit("The distance must be greater than zero.")
    mm_per_pixel = distance_mm / distance_px
    com(lambda: raster.ScaleEntity(point(0, 0), mm_per_pixel))
    com(lambda: raster.Update())
    print(f"Calibrated: {distance_px:.1f} px = {distance_mm:.1f} mm -> {mm_per_pixel:.5f} mm/px "
          f"(reuse with --mm-per-pixel {mm_per_pixel:.5f})")
    return mm_per_pixel


def send_to_back(document, entity) -> None:
    """Put the raster below every crack line (ActiveX lacks a portable draw-order API)."""
    handle = str(com(lambda: entity.Handle))
    com(lambda: document.SendCommand(
        f'(progn (setq crackimg (handent "{handle}")) '
        '(if crackimg (command "_.DRAWORDER" crackimg "" "_Back")) (princ)) \n'
    ))


def draw_cracks(document, cracks, text_height: float, label_min_length: float, summary_lines,
                image_width_mm: float, image_height_mm: float) -> tuple[int, int]:
    import pythoncom
    from win32com.client import VARIANT

    model_space = com(lambda: document.ModelSpace)
    com(lambda: setattr(document, "ActiveLayer", document.Layers.Item("CRACKS")))
    drawn = 0
    for crack in cracks:
        for polyline in crack["polylines"]:
            flat = [value for xy in polyline["points"] for value in xy]
            coordinates = VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, flat)
            com(lambda: model_space.AddLightWeightPolyline(coordinates))
            drawn += 1
    com(lambda: setattr(document, "ActiveLayer", document.Layers.Item("CRACK_LABELS")))
    labelled = 0
    for crack in cracks:
        if crack["length"] < label_min_length:
            continue
        x, y = crack["anchor"]
        text = f"{crack['name']}  L = {crack['length']:.1f} mm"
        # Just right of the crack's midpoint, vertically centred on it.
        com(lambda: model_space.AddText(text, point(x + 0.6 * text_height, y - 0.5 * text_height), text_height))
        labelled += 1
    if summary_lines:
        # MText hangs from its insertion point, so start high enough to clear the image.
        note_height = 1.5 * text_height
        top = image_height_mm + (len(summary_lines) * 1.7 + 2) * note_height
        note = com(lambda: model_space.AddMText(point(0, top), image_width_mm, "\\P".join(summary_lines)))
        com(lambda: setattr(note, "Height", note_height))
    return drawn, labelled


# --------------------------------------------------------------------------- #
def add_autocad_arguments(parser) -> None:
    """AutoCAD options; hairline_crack_detection.py uses them too, to hand over directly."""
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--mm-per-pixel", type=float,
                       help="Real size of one image pixel in mm (default: 1, uncalibrated image pixel scale)")
    scale.add_argument("--image-width-mm", type=float, help="Real width of the whole image in mm")
    parser.add_argument("--start-autocad", action="store_true", help="Start AutoCAD without asking if it is closed")
    parser.add_argument("--new-drawing", action="store_true",
                        help="Draw in a new metric drawing (saved to the output folder) instead of the active one")
    parser.add_argument("--text-height", type=float, help="Label height in mm (default: 0.6%% of the image height)")
    parser.add_argument("--label-min-length", type=float,
                        help="Only label cracks at least this long, in mm; 0 labels all. Default: 5 x the "
                             "text height, since a label longer than its crack only adds clutter "
                             "(every crack is still drawn and listed in crack_lengths_mm.csv)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog="Any other option is passed to hairline_crack_detection.py.")
    parser.add_argument("image", nargs="?", type=Path, help="Input image; omit to choose in a dialog")
    parser.add_argument("--output", type=Path, help="Detection folder (default: <image>_crack_detection)")
    parser.add_argument("--redetect", action="store_true", help="Run detection even if results exist")
    add_autocad_arguments(parser)
    return parser.parse_known_args(argv)


def main(argv=None):
    args, detection_options = parse_args(argv)
    if args.image is None:
        from hairline_crack_detection import choose_image
        args.image = choose_image()
    image_path = args.image.resolve()
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")
    output = (args.output or image_path.with_name(f"{image_path.stem}_crack_detection")).resolve()

    # 1. Detection (or reuse of earlier results)
    if args.redetect or not (output / "crack_polylines_rdp.csv").is_file():
        import hairline_crack_detection as detector

        detector.main(detector.parse_args([str(image_path), "--output", str(output), *detection_options]))
    else:
        print(f"Reusing detection results in {output} (use --redetect to run detection again).")
        if detection_options:
            print(f"  Ignored detection options {detection_options}: add --redetect to apply them.")
    draw_in_autocad(image_path, output, args)


def draw_in_autocad(image_path: Path, output: Path, args) -> None:
    """Draw the detection results in `output` over the image in AutoCAD, in millimetres.

    `args` needs the options from add_autocad_arguments(). Called by main() and
    directly by hairline_crack_detection.py once detection has finished.
    """
    print("\nDrawing the cracks in AutoCAD ...")
    width_px, height_px = image_size(image_path)
    polylines, skeleton_px = read_polylines(output)
    if not polylines:
        raise SystemExit("No cracks were detected; nothing to draw.")

    pixel_scale = args.mm_per_pixel is None and args.image_width_mm is None
    mm_per_pixel = (args.mm_per_pixel if args.mm_per_pixel is not None else
                    args.image_width_mm / width_px if args.image_width_mm is not None else 1.0)
    if not math.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
        raise SystemExit("The scale must be finite and greater than zero.")
    if pixel_scale:
        print("Using image pixel scale: 1 pixel = 1 drawing unit (nominally 1 mm). "
              "Lengths are uncalibrated, not physical measurements.")
    # Save lengths even if AutoCAD cannot be reached.
    cracks = build_cracks(polylines, skeleton_px, height_px, mm_per_pixel)
    write_csvs(output, cracks, mm_per_pixel)

    # 2-3. AutoCAD session, drawing and millimetre units
    application, _ = connect_autocad(args.start_autocad)
    document, created = open_drawing(application, args.new_drawing)
    set_millimetres(document)
    for name, (colour, lineweight) in LAYERS.items():
        ensure_layer(document, name, colour, lineweight)

    # 4. Image at the chosen scale; no interactive point picking is needed.
    print(f"Inserting the {width_px} x {height_px} px image; AutoCAD may be busy loading it for a while ...")
    raster = insert_image(document, image_path, width_px, height_px, mm_per_pixel)
    wait_until_idle(application)

    # 5. Cracks, lengths and labels
    text_height = args.text_height or max(1.0, round(0.006 * height_px * mm_per_pixel, 1))
    label_min_length = args.label_min_length if args.label_min_length is not None else 5 * text_height
    total = sum(crack["length"] for crack in cracks)
    summary_lines = [
        f"CRACK DETECTION - {image_path.name}",
        f"Scale {mm_per_pixel:.5f} mm/px; image {width_px} x {height_px} px = "
        f"{width_px * mm_per_pixel:,.0f} x {height_px * mm_per_pixel:,.0f} mm",
        f"{len(cracks)} cracks, total length {total:,.1f} mm (measured along the drawn polylines)",
        f"Labelled: cracks >= {label_min_length:,.1f} mm; all lengths in crack_lengths_mm.csv",
    ]
    if pixel_scale:
        summary_lines.insert(1, "UNCALIBRATED IMAGE SCALE: 1 pixel = 1 drawing unit; mm labels are nominal")
    drawn, labelled = draw_cracks(document, cracks, text_height, label_min_length, summary_lines,
                                  width_px * mm_per_pixel, height_px * mm_per_pixel)
    send_to_back(document, raster)
    com(lambda: document.Regen(1))  # all viewports
    com(lambda: application.ZoomExtents())

    if created:
        # Never overwrite an earlier drawing (it may also be open in AutoCAD right now).
        dwg, number = output / f"{image_path.stem}_cracks_mm.dwg", 1
        while dwg.exists():
            number += 1
            dwg = output / f"{image_path.stem}_cracks_mm_{number}.dwg"
        try:
            com(lambda: document.SaveAs(str(dwg)))
            print(f"Saved the new drawing: {dwg}")
        except Exception as exc:
            print(f"The new drawing is open in AutoCAD but could not be saved as {dwg} ({exc}); "
                  "save it from AutoCAD.")
    else:
        print("The active drawing was modified but not saved; save it in AutoCAD when you are satisfied.")
    print(f"Drawn {drawn} polylines for {len(cracks)} cracks on layer CRACKS, {labelled} length labels "
          f"(text height {text_height} mm, cracks >= {label_min_length:.1f} mm; --label-min-length 0 "
          f"labels all) on CRACK_LABELS; total crack length {total:,.1f} mm.")
    print("Longest cracks:")
    for crack in sorted(cracks, key=lambda c: c["length"], reverse=True)[:10]:
        print(f"  {crack['name']:>6s}  {crack['length']:9.1f} mm")
    print(f"Lengths per crack: {output / 'crack_lengths_mm.csv'}")


if __name__ == "__main__":
    main()
