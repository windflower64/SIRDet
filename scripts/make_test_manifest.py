"""Record test membership and visible-label hashes without redistributing images."""

import argparse
import csv
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/test_pairs.csv")
    args = parser.parse_args()
    root = args.data_root.resolve()
    rows = []
    for image in sorted((root / "test/visible/images").iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        ir = root / "test/infrared/images" / image.name
        label = root / "test/visible/labels" / (image.stem + ".txt")
        if not ir.is_file() or not label.is_file():
            raise FileNotFoundError(f"Incomplete pair: {image.name}")
        raw = label.read_bytes()
        rows.append((image.relative_to(root).as_posix(), ir.relative_to(root).as_posix(),
                     label.relative_to(root).as_posix(), hashlib.sha256(raw).hexdigest(),
                     sum(bool(line.strip()) for line in raw.decode("utf-8-sig").splitlines())))
    if not rows:
        raise ValueError("No test pairs found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("visible", "infrared", "visible_label", "visible_label_sha256", "boxes"))
        writer.writerows(rows)
    print(f"Saved {len(rows)} pairs, {sum(row[-1] for row in rows)} boxes: {args.output}")


if __name__ == "__main__":
    main()
