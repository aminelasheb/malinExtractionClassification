import os
import json
import shutil
import argparse
import pandas as pd
from pathlib import Path
import sys
import re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

LABEL_DICT = {
    'Associe': 0, 'AssocieCoche': 1, 'CM': 2, 'CacheIntrus': 3, 'Classe': 4,
    'ClasseCM': 5, 'CliqueEcrire': 6, 'CocheGroupeMots': 7, 'CocheIntrus': 8,
    'CocheLettre': 9, 'CocheMot': 10, 'CocheMot*': 11, 'CochePhrase': 12,
    'Echange': 13, 'EditPhrase': 14, 'EditTexte': 15, 'ExpressionEcrite': 16,
    'GenreNombre': 17, 'Phrases': 18, 'Question': 19, 'RC': 20, 'RCCadre': 21,
    'RCDouble': 22, 'RCImage': 23, 'Texte': 24, 'Trait': 25,
    'TransformeMot': 26, 'TransformePhrase': 27, 'VraiFaux': 28
}


def get_folder_name(label):
    return "CocheMotEtoile" if label == "CocheMot*" else label


def convert_id(old_id):
    match = re.match(r"p(\d+)_ex(\d+)", old_id)
    if match:
        return f"P{match.group(1)}Ex{match.group(2)}"

    match_special = re.match(r"p(\d+)_(.+)", old_id)
    if match_special:
        return f"P{match_special.group(1)}Ex-{match_special.group(2)}"

    return old_id


def organize(pdf_name):

    base_dir = Path(__file__).resolve().parent

    extraction_dir = base_dir / "extractionOut"
    extraction_style_dir = base_dir / "extractionOutStyle"
    classification_dir = base_dir / "classificationOut"

    output_root = base_dir / "SORTIES" / pdf_name

    out_extraction = output_root / "Extraction_exercices"
    out_extraction_style = output_root / "Extraction_exercices --style"
    out_classification = output_root / "CategorisationExercices"
    out_classification_style = output_root / "CategorisationExercices --style"

    for folder in [
        out_extraction,
        out_extraction_style,
        out_classification,
        out_classification_style,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    for label in LABEL_DICT.keys():
        (out_classification / get_folder_name(label)).mkdir(exist_ok=True)
        (out_classification_style / get_folder_name(label)).mkdir(exist_ok=True)

    tsv_files = list(classification_dir.glob("pred_page_*.tsv"))

    total_exercises = 0
    matched_exercises = 0

    for tsv_path in tsv_files:

        page_name = tsv_path.stem.replace("pred_", "")
        json_path = extraction_dir / f"{page_name}.json"
        json_style_path = extraction_style_dir / f"{page_name}--style.json"

        if not json_path.exists():
            continue

        df = pd.read_csv(tsv_path, sep="\t")
        pred_map = dict(zip(df["id"], df["pred"]))

        # -------- NORMAL PAGE --------
        with open(json_path, "r", encoding="utf-8") as f:
            exercises = json.load(f)

        id_mapping = {}

        for ex in exercises:
            old_id = ex["id"]
            new_id = convert_id(old_id)
            id_mapping[old_id] = new_id
            ex["id"] = new_id
            total_exercises += 1

        with open(out_extraction / f"{page_name}.json", "w", encoding="utf-8") as f:
            json.dump(exercises, f, indent=4, ensure_ascii=False)

        # -------- STYLE PAGE --------
        exercises_style = {}
        if json_style_path.exists():
            with open(json_style_path, "r", encoding="utf-8") as f:
                style_data = json.load(f)

            for ex in style_data:
                old_id = ex["id"]
                if old_id in id_mapping:
                    ex["id"] = id_mapping[old_id]
                    exercises_style[old_id] = ex

            with open(out_extraction_style / f"{page_name}.json", "w", encoding="utf-8") as f:
                json.dump(style_data, f, indent=4, ensure_ascii=False)

        # -------- CLASSIFICATION --------
        for old_id, new_id in id_mapping.items():

            if old_id not in pred_map:
                print(f"[MISS] Pas de prédiction pour {old_id}")
                continue

            label = pred_map[old_id]

            if label not in LABEL_DICT:
                print(f"[WARN] Label inconnu {label}")
                continue

            folder = get_folder_name(label)

            # Normal
            ex_obj = next(e for e in exercises if e["id"] == new_id)
            with open(out_classification / folder / f"{new_id}.json", "w", encoding="utf-8") as f:
                json.dump(ex_obj, f, indent=4, ensure_ascii=False)

            # Style
            if old_id in exercises_style:
                with open(out_classification_style / folder / f"{new_id}.json", "w", encoding="utf-8") as f:
                    json.dump(exercises_style[old_id], f, indent=4, ensure_ascii=False)

            matched_exercises += 1

    print("\n========== RÉSUMÉ ==========")
    print(f"Total exercices trouvés : {total_exercises}")
    print(f"Total classés : {matched_exercises}")
    print("=============================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_name")
    args = parser.parse_args()
    organize(args.pdf_name)
