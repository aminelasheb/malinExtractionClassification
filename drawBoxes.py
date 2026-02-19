import os
import json
import cv2
from pathlib import Path

# Base directory
base_dir = Path(__file__).parent.resolve()

images_path = base_dir / "files"
json_path = base_dir / "output" / "detImages" / "predict"
output_path = base_dir / "files-out"

# Create output folder if not exists
os.makedirs(output_path, exist_ok=True)

# 1. On récupère la liste de TOUTES les images sources
valid_extensions = (".png", ".jpg", ".jpeg")
all_images = [f for f in os.listdir(images_path) if f.lower().endswith(valid_extensions)]

print(f"[INFO] Traitement de {len(all_images)} images depuis {images_path}")

# 2. On itère sur les IMAGES (et non plus sur les JSONs)
for image_name in all_images:
    image_file = os.path.join(images_path, image_name)

    # Chargement de l'image
    img = cv2.imread(image_file)
    if img is None:
        print(f"[WARN] Could not load {image_file}")
        continue

    # 3. On cherche s'il existe un JSON correspondant
    json_name = os.path.splitext(image_name)[0] + ".json"
    json_file_path = os.path.join(json_path, json_name)

    if os.path.exists(json_file_path):
        # --- CAS 1 : Il y a des détections (JSON trouvé) ---
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract page number for labeling
            page_num = ''.join([c for c in image_name if c.isdigit()]) or "0"

            for shape in data.get("shapes", []):
                pts = shape["points"]
                x1, y1 = map(int, pts[0])
                x2, y2 = map(int, pts[1])
                shape_id = shape["id"]

                # --- Draw Box (Red) ---
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 6)

                # --- Draw Label ---
                label_text = f"p{page_num}c{shape_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 2.8
                thickness = 8

                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, scale, thickness)
                box_cx = (x1 + x2) // 2
                box_cy = (y1 + y2) // 2

                text_x = box_cx - text_w // 2
                text_y = box_cy + text_h // 2
                padding = 20

                tx1 = text_x - padding
                ty1 = text_y - text_h - padding
                tx2 = text_x + text_w + padding
                ty2 = text_y + padding

                # Background for text
                overlay = img.copy()
                cv2.rectangle(overlay, (tx1, ty1), (tx2, ty2), (0, 0, 0), -1)
                alpha = 0.85
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                # Text itself
                cv2.putText(img, label_text, (text_x, text_y), font,
                            scale, (255, 255, 255), thickness, cv2.LINE_AA)

            print(f"[OK] Detections drawn for {image_name}")

        except Exception as e:
            print(f"[ERR] Error reading JSON for {image_name}: {e}")
            # En cas d'erreur JSON, on sauvegarde quand même l'image originale
    else:
        # --- CAS 2 : Pas de JSON (Pas de détections) ---
        print(f"[INFO] No crops/json for {image_name}. Saving original.")

    # 4. Sauvegarde finale (Modifiée ou Originale) dans files-out
    out_file = os.path.join(output_path, image_name)
    cv2.imwrite(out_file, img)

print("[DONE] DrawBoxes finished.")