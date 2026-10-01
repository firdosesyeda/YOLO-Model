from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SOURCE_PERSON_ID = 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy Construction-PPE person boxes into a one-class YOLO dataset.")
    parser.add_argument("--source", type=Path, default=Path("construction-ppe"))
    parser.add_argument("--output", type=Path, default=Path("construction_person"))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise SystemExit("Output must be outside the source dataset directory.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty; refusing to overwrite: {output}")

    total_images = 0
    for split in SPLITS:
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        destination_images = output / "images" / split
        destination_labels = output / "labels" / split
        if not source_images.is_dir() or not source_labels.is_dir():
            raise SystemExit(f"Missing images or labels for split '{split}' under {source}")
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        person_boxes = 0
        images = sorted(
            path for path in source_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for image_path in images:
            shutil.copy2(image_path, destination_images / image_path.name)
            source_label = source_labels / f"{image_path.stem}.txt"
            person_lines: list[str] = []
            if source_label.is_file():
                for line_number, line in enumerate(source_label.read_text(encoding="utf-8-sig").splitlines(), 1):
                    columns = line.split()
                    if not columns:
                        continue
                    try:
                        class_id = int(columns[0])
                    except ValueError as error:
                        raise SystemExit(f"Invalid class ID in {source_label}:{line_number}") from error
                    if class_id != SOURCE_PERSON_ID:
                        continue
                    if len(columns) != 5:
                        raise SystemExit(f"Expected a YOLO detection row with 5 columns in {source_label}:{line_number}")
                    try:
                        coordinates = [float(value) for value in columns[1:]]
                    except ValueError as error:
                        raise SystemExit(f"Invalid coordinates in {source_label}:{line_number}") from error
                    if any(value < 0.0 or value > 1.0 for value in coordinates):
                        raise SystemExit(f"Coordinates outside [0, 1] in {source_label}:{line_number}")
                    person_lines.append("0 " + " ".join(columns[1:]))
                    person_boxes += 1
            destination_label = destination_labels / f"{image_path.stem}.txt"
            destination_label.write_text("\n".join(person_lines) + ("\n" if person_lines else ""), encoding="utf-8")
        total_images += len(images)
        print(f"{split}: {len(images)} images; person boxes={person_boxes}")

    config = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["person"],
    }
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source_license = source / "LICENSE"
    if source_license.is_file():
        shutil.copy2(source_license, output / source_license.name)
    print(f"Prepared {total_images} images at {output}; source dataset was not modified.")


if __name__ == "__main__":
    main()