from ultralytics import YOLO
from pathlib import Path
import cv2
import json

# Base directory = dossier du script
base_dir = Path(__file__).resolve().parent

# Directories
files_dir = base_dir / "files"
models_dir = base_dir / "models"
output_dir = base_dir / "output"
output_dir.mkdir(exist_ok=True)

# Classes per model
classes_dict = {
    "detImages": ["image"],
}

# Get all PNG files
images = list(files_dir.glob("*.png"))

# Model paths (RELATIVE)
model_paths = {
    "detImages": models_dir / "detImages.pt",
}

run_folders = []

# Step 1: Detection
for model_name, model_path in model_paths.items():
    print(f"\n=== Loading model {model_name} ===")
    model = YOLO(str(model_path))

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(exist_ok=True, parents=True)

    for image_path in images:
        print(f"Processing {image_path.name} with {model_name}...")
        _ = model.predict(
            source=str(image_path),
            save=True,
            save_txt=True,
            project=str(model_output_dir),
            name="predict",
            exist_ok=True
        )

    run_folders.append((model_name, model_output_dir))
    print(f"==> {model_name} finished\n")

# Step 2: TXT → JSON
for model_name, run_folder in run_folders:
    labels_dir = run_folder / "predict" / "labels"
    images_dir = run_folder / "predict"

    if not labels_dir.exists():
        print(f"No labels found for {model_name}")
        continue

    print(f"Transforming labels to JSON for {model_name}")

    for txt_file in labels_dir.glob("*.txt"):

        image_path = images_dir / f"{txt_file.stem}.jpg"
        if not image_path.exists():
            image_path = images_dir / f"{txt_file.stem}.png"
        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        h, w = img.shape[:2]

        shapes = []
        with open(txt_file, "r") as f:
            for idx, line in enumerate(f.readlines()):
                cls, x_c, y_c, bw, bh = map(float, line.strip().split())
                cls = int(cls)

                x_center = x_c * w
                y_center = y_c * h
                width = bw * w
                height = bh * h

                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2

                shape = {
                    "id": idx,
                    "label": classes_dict.get(model_name, [str(cls)])[cls],
                    "points": [
                        [x_min, y_min],
                        [x_max, y_max]
                    ]
                }
                shapes.append(shape)

        json_dict = {
            "shapes": shapes,
            "imageHeight": h,
            "imageWidth": w
        }

        json_path = images_dir / f"{txt_file.stem}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(json_dict, jf, ensure_ascii=False, indent=2)

        print(f"[OK] Saved JSON: {json_path}")
